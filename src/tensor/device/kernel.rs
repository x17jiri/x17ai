//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::sync::Arc;

use crate::ErrPack;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut, check_borrows};
use crate::tensor::dim_merger::{DimMerger, MergedDim};
use crate::tensor::generic::map::ND;
use crate::tensor::{Tensor, TensorOpError, generic};
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

pub struct VecArg {
	pub index: usize,
	pub name: String,
}

//--------------------------------------------------------------------------------------------------

pub enum ScalarExpr {
	ElemArg(Arc<ElemArg>),
	ConstArg(Arc<ConstArg>),
	FloatLiteral(FloatLiteral),

	DotExpr(Arc<VecArg>, Arc<VecArg>),

	AddExpr(Arc<ScalarExpr>, Arc<ScalarExpr>),
}

//--------------------------------------------------------------------------------------------------

pub struct KernelData {
	pub id: usize,
	pub name: String,
	pub elem_args: Box<[Arc<ElemArg>]>,
	pub vec_args: Box<[Arc<VecArg>]>,
	pub const_args: Box<[Arc<ConstArg>]>,
	pub expr: Arc<ScalarExpr>,
}

//--------------------------------------------------------------------------------------------------

pub struct Kernel<const E: usize, const V: usize, const C: usize> {
	data: Arc<KernelData>,
}

impl<const E: usize, const V: usize, const C: usize> Kernel<E, V, C> {
	pub fn new(data: Arc<KernelData>) -> Self {
		debug_assert!(data.elem_args.len() == E);
		debug_assert!(data.vec_args.len() == V);
		debug_assert!(data.const_args.len() == C);
		Self { data }
	}

	pub fn call(
		&self,
		output: &Tensor,
		elem_args: [&Tensor; E],
		vec_args: [&Tensor; V],
		const_args: [f64; C],
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); 1 + E + V]:,
	{
		let output_dims = output.map().dims.as_slice();
		let elem_dims = elem_args.map(|t| t.map().dims.as_slice());
		let Some(vec_dims) = vec_args.try_map(|t| t.map().dims.as_slice().split_last()) else {
			cold_path();
			return Err(TensorOpError::missing_vec_dimension());
		};
		let vec_features = vec_dims.map(|(&feature_dim, _)| feature_dim);
		let vec_dims = vec_dims.map(|(_, dim)| dim);
		let all_dims = array::concat_arrays([output_dims], elem_dims);
		let all_dims = array::concat_arrays(all_dims, vec_dims);

		let merged: [MergedDim<{ 1 + E + V }>; 2] = DimMerger::merge(all_dims)?;

		unsafe {
			let mut m_fail = 0;
			let mut o_buffer = DeviceBufferRefMut::new_unsafe(output.buf().as_ref(), &mut m_fail);
			let mut o_device_data = o_buffer.device_buffer().device_data;

			let mut c_fail = 0;
			let elem_buffers: [DeviceBufferRef; E] = std::array::from_fn(|i| {
				DeviceBufferRef::new_unsafe(elem_args[i].buf().as_ref(), &mut c_fail)
			});

			// TODO - the `map` destroys the `DeviceBufferRef`
			// and so the `elem_device_data` is not safe
			let elem_device_data = elem_buffers.map(|buf| buf.device_buffer().device_data);
			std::mem::drop(elem_buffers);

			let vec_tensors: [generic::Tensor<ND<3>, DeviceBufferRef>; V] =
				std::array::from_fn(|i| {
					generic::Tensor::new_unchecked(
						ND {
							dims: [
								merged[0].get(1 + E + i),
								merged[1].get(1 + E + i),
								vec_features[i],
							],
							offset: vec_args[i].map().offset,
						},
						DeviceBufferRef::new_unsafe(vec_args[i].buf().as_ref(), &mut c_fail),
					)
				});

			check_borrows(c_fail, m_fail)?;

			//-------------

			let mut m_fail = 0;
			let mut o_tensor = generic::Tensor::new_unchecked(
				ND {
					dims: [merged[0].get(0), merged[1].get(0)],
					offset: output.map().offset,
				},
				DeviceBufferRefMut::new_unsafe(output.buf().as_ref(), &mut m_fail),
			);

			let mut c_fail = 0;
			let elem_tensors: [Option<generic::Tensor<ND<2>, DeviceBufferRef>>; E] =
				std::array::from_fn(|i| {
					Some(generic::Tensor::new_unchecked(
						ND {
							dims: [merged[0].get(1 + i), merged[1].get(1 + i)],
							offset: elem_args[i].map().offset,
						},
						DeviceBufferRef::new_unsafe(elem_args[i].buf().as_ref(), &mut c_fail),
					))
				});
			let vec_tensors: [generic::Tensor<ND<3>, DeviceBufferRef>; V] =
				std::array::from_fn(|i| {
					generic::Tensor::new_unchecked(
						ND {
							dims: [
								merged[0].get(1 + E + i),
								merged[1].get(1 + E + i),
								vec_features[i],
							],
							offset: vec_args[i].map().offset,
						},
						DeviceBufferRef::new_unsafe(vec_args[i].buf().as_ref(), &mut c_fail),
					)
				});

			check_borrows(c_fail, m_fail)?;

			// TODO - ensure_safe
			// TODO - ensure all on same device
			// TODO - other things may need to be checked before running the kernel

			/*output.executor().run_kernel(
				self.data.as_ref(),
				&mut o_tensor,
				&elem_tensors,
				&vec_tensors,
				&const_args,
			)?;*/
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
