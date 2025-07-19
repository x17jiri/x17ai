//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::{cold_path, likely};
use std::sync::Arc;

use crate::ErrPack;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut, check_borrows};
use crate::tensor::device::executor::{KernelElemArg, KernelOutput, KernelReduceArg};
use crate::tensor::dim_merger::{DimMerger, MergedDim};
use crate::tensor::generic::map::SizeAndStride;
use crate::tensor::{Tensor, TensorOpError};
use crate::util::array;

//--------------------------------------------------------------------------------------------------

pub struct ElemArg {
	pub index: usize,
	pub name: String,
}

pub struct ConstArg {
	pub index: usize,
	pub name: String,
}

pub struct FloatLiteral {
	pub value: f64,
}

pub struct ReduceArg {
	pub index: usize,
	pub name: String,
}

//--------------------------------------------------------------------------------------------------

pub enum ScalarExpr {
	ElemArg(Arc<ElemArg>),
	ConstArg(Arc<ConstArg>),
	FloatLiteral(FloatLiteral),

	DotExpr(Arc<ReduceArg>, Arc<ReduceArg>),

	SqrtExpr(Arc<ScalarExpr>),
	RecipExpr(Arc<ScalarExpr>, Arc<ScalarExpr>),
	AddExpr(Arc<ScalarExpr>, Arc<ScalarExpr>),
	MulExpr(Arc<ScalarExpr>, Arc<ScalarExpr>),
}

//--------------------------------------------------------------------------------------------------

pub struct KernelData {
	pub id: usize,
	pub name: String,
	pub elem_args: Box<[Arc<ElemArg>]>,
	pub reduce_args: Box<[Arc<ReduceArg>]>,
	pub const_args: Box<[Arc<ConstArg>]>,
	pub expr: Arc<ScalarExpr>,
}

//--------------------------------------------------------------------------------------------------

pub struct Kernel<const E: usize, const R: usize, const C: usize> {
	data: Arc<KernelData>,
}

impl<const E: usize, const R: usize, const C: usize> Kernel<E, R, C> {
	pub fn new(data: Arc<KernelData>) -> Self {
		debug_assert!(data.elem_args.len() == E);
		debug_assert!(data.reduce_args.len() == R);
		debug_assert!(data.const_args.len() == C);
		Self { data }
	}

	#[allow(clippy::too_many_lines)]
	#[allow(clippy::indexing_slicing)]
	pub fn run(
		&self,
		output: &Tensor,
		elem_args: [&Tensor; E],
		reduce_args: [&Tensor; R],
		const_args: [f64; C],
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); 1 + E + R]:,
	{
		if R > 0 {
			let output_dims = output.map().dims.as_slice().split_last();
			let elem_args_dims = elem_args.try_map(|t| t.map().dims.as_slice().split_last());
			let reduce_args_dims = reduce_args.try_map(|t| t.map().dims.as_slice().split_last());

			let (Some(output_dims), Some(elem_dims), Some(reduce_dims)) =
				(output_dims, elem_args_dims, reduce_args_dims)
			else {
				cold_path();
				return Err(TensorOpError::missing_reduce_dimension());
			};

			let (output_top_dim, output_batch_dims) = output_dims;
			let elem_args_top_dim = elem_dims.map(|(&top_dim, _)| top_dim);
			let elem_args_batch_dims = elem_dims.map(|(_, batch_dim)| batch_dim);
			let reduce_args_top_dim = reduce_dims.map(|(&top_dim, _)| top_dim);
			let reduce_args_batch_dims = reduce_dims.map(|(_, batch_dim)| batch_dim);

			if output_top_dim.size != 1 || elem_args_top_dim.iter().any(|dim| dim.size != 1) {
				cold_path();
				// we would have to broadcast the result of the reduction
				// TODO - maybe use a different error
				return Err(TensorOpError::invalid_shape());
			}
			if reduce_args_top_dim.iter().any(|vec| vec.stride != 1) {
				cold_path();
				return Err(TensorOpError::not_contiguous());
			}

			let all_batch_dims = array::concat_arrays([output_batch_dims], elem_args_batch_dims);
			let all_batch_dims = array::concat_arrays(all_batch_dims, reduce_args_batch_dims);

			let merged: [MergedDim<{ 1 + E + R }>; 2] = DimMerger::merge(all_batch_dims)?;

			let reduce_inp: [KernelReduceArg; R] = std::array::from_fn(|i| {
				let arg = reduce_args[i];
				KernelReduceArg {
					reduction_size: reduce_args_top_dim[i].size,
					stride: [merged[0].strides[1 + E + i], merged[1].strides[1 + E + i]],
					offset: arg.map().offset,
					device_data: arg.buf().device_data,
				}
			});

			let inp: [KernelElemArg; E] = std::array::from_fn(|i| {
				let arg = elem_args[i];
				KernelElemArg {
					stride: [merged[0].strides[1 + i], merged[1].strides[1 + i]],
					offset: arg.map().offset,
					device_data: arg.buf().device_data,
				}
			});

			let out = [KernelOutput {
				size: [merged[0].size, merged[1].size],
				stride: [merged[0].strides[0], merged[1].strides[0]],
				offset: output.map().offset,
				device_data: output.buf().device_data,
			}];

			unsafe {
				let mut c_fail = 0;
				let reduce_borrows: [DeviceBufferRef; R] = std::array::from_fn(|i| {
					let arg = &reduce_args[i];
					DeviceBufferRef::new_unsafe(arg.buf().as_ref(), &mut c_fail)
				});
				let elem_borrows: [Option<DeviceBufferRef>; E] = std::array::from_fn(|i| {
					let arg = &elem_args[i];
					let same_as_output = std::ptr::eq(arg.buf().as_ref(), output.buf().as_ref())
						&& likely(inp[i].offset == out[0].offset && inp[i].stride == out[0].stride);
					if same_as_output {
						None
					} else {
						Some(DeviceBufferRef::new_unsafe(arg.buf().as_ref(), &mut c_fail))
					}
				});

				let mut m_fail = 0;
				let out_borrow = DeviceBufferRefMut::new_unsafe(output.buf().as_ref(), &mut m_fail);

				check_borrows(c_fail, m_fail)?;

				// TODO - ensure_safe
				// TODO - ensure all on same device
				// TODO - other things may need to be checked before running the kernel

				output.executor().run_kernel(
					self.data.as_ref(),
					out.as_ptr(),
					inp.as_ptr(),
					reduce_inp.as_ptr(),
					const_args.as_ptr(),
				)?;

				std::mem::drop(out_borrow);
				std::mem::drop(elem_borrows);
				std::mem::drop(reduce_borrows);
			}
		} else {
			let output_dims: &[SizeAndStride] = output.map().dims.as_slice();
			let elem_args_dims: [&[SizeAndStride]; E] = elem_args.map(|t| t.map().dims.as_slice());

			let all_dims = array::concat_arrays([output_dims], elem_args_dims);

			let merged = DimMerger::merge::<2>(all_dims)?;

			let inp: [KernelElemArg; E] = std::array::from_fn(|i| {
				let elem_arg = elem_args[i];
				KernelElemArg {
					stride: [merged[0].strides[1 + i], merged[1].strides[1 + i]],
					offset: elem_arg.map().offset,
					device_data: elem_arg.buf().device_data,
				}
			});

			let out = [KernelOutput {
				size: [merged[0].size, merged[1].size],
				stride: [merged[0].strides[0], merged[1].strides[0]],
				offset: output.map().offset,
				device_data: output.buf().device_data,
			}];

			unsafe {
				let mut c_fail = 0;
				let elem_borrows: [Option<DeviceBufferRef>; E] = std::array::from_fn(|i| {
					let arg = &elem_args[i];
					let same_as_output = std::ptr::eq(arg.buf().as_ref(), output.buf().as_ref())
						&& likely(inp[i].offset == out[0].offset && inp[i].stride == out[0].stride);
					if same_as_output {
						None
					} else {
						Some(DeviceBufferRef::new_unsafe(arg.buf().as_ref(), &mut c_fail))
					}
				});

				let mut m_fail = 0;
				let out_borrow = DeviceBufferRefMut::new_unsafe(output.buf().as_ref(), &mut m_fail);

				check_borrows(c_fail, m_fail)?;

				// TODO - ensure_safe
				// TODO - ensure all on same device
				// TODO - other things may need to be checked before running the kernel
				let reduce_inp = [];

				output.executor().run_kernel(
					self.data.as_ref(),
					out.as_ptr(),
					inp.as_ptr(),
					reduce_inp.as_ptr(),
					const_args.as_ptr(),
				)?;

				std::mem::drop(out_borrow);
				std::mem::drop(elem_borrows);
			}
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
