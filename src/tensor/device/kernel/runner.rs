//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::UnsafeCell;
use std::hint::{cold_path, likely};
use std::mem::MaybeUninit;
use std::sync::{Arc, RwLock};

use crate::ErrPack;
use crate::tensor::device::buffer::{
	DeviceBufferRef, DeviceBufferRefMut, KernelElemArg, KernelOutput, KernelReduceArg,
	check_borrows,
};
use crate::tensor::device::kernel::expr::{DynExpr, Expr, ExprDiscriminant, ExprToDyn};
use crate::tensor::device::kernel::registry::{KernelMap, KernelRegistry};
use crate::tensor::dim_merger::DimMerger;
use crate::tensor::generic::map::SizeAndStride;
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

pub const KEY_BATCH_SIZE: usize =
	std::mem::size_of::<u64>() / std::mem::size_of::<ExprDiscriminant>();

union KeyUnion<const PADDED_KEY_LEN: usize, const BATCHED_KEY_LEN: usize>
where
	[(); PADDED_KEY_LEN]:,
	[(); BATCHED_KEY_LEN]:,
{
	discriminants: [ExprDiscriminant; PADDED_KEY_LEN],
	key: [u64; BATCHED_KEY_LEN],
}

//--------------------------------------------------------------------------------------------------

pub struct KernelData {
	pub id: usize,
	pub key: Box<[u64]>,
	pub expr: Arc<DynExpr>,
	pub elemwise_count: usize,
	pub reduce_count: usize,
	pub scalar_count: usize,
}

pub struct KernelRunner {
	registry: Arc<RwLock<KernelRegistry>>,
	cache: UnsafeCell<KernelMap>,
}

impl Default for KernelRunner {
	fn default() -> Self {
		Self::new()
	}
}

impl KernelRunner {
	pub fn new() -> Self {
		Self {
			registry: KernelRegistry::instance(),
			cache: UnsafeCell::new(KernelMap::new()),
		}
	}

	#[inline]
	#[allow(clippy::panic_in_result_fn)]
	pub fn run<E: const Expr + ExprToDyn>(
		&self,
		output: &Tensor,
		expr: E,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); 1 - E::REDUCE_OP_COUNT]:, // REDUCE_OP_COUNT <= 1
		[(); E::ELEMWISE_COUNT]:,
		[(); E::REDUCE_COUNT]:,
		[(); E::SCALAR_COUNT]:,
		[(); E::PADDED_KEY_LEN]:,
		[(); E::BATCHED_KEY_LEN]:,
		[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:,
	{
		if E::CONST {
			let mut elem_args = [MaybeUninit::uninit(); E::ELEMWISE_COUNT];
			let cnt = expr.elemwise_tensors(&mut elem_args, 0);
			assert!(cnt == E::ELEMWISE_COUNT);
			let elem_args = unsafe { MaybeUninit::array_assume_init(elem_args) };

			let mut reduce_args = [MaybeUninit::uninit(); E::REDUCE_COUNT];
			let cnt = expr.reduce_tensors(&mut reduce_args, 0);
			assert!(cnt == E::REDUCE_COUNT);
			let reduce_args = unsafe { MaybeUninit::array_assume_init(reduce_args) };

			let mut scalar_args = [0_f64; E::SCALAR_COUNT];
			let cnt = expr.scalars(&mut scalar_args, 0);
			assert!(cnt == E::SCALAR_COUNT);
			let scalar_args = scalar_args;

			self.run_const(output, elem_args, reduce_args, scalar_args)
		} else {
			todo!("non-constant exprs not implemented yet");
		}
	}

	#[inline(never)]
	#[allow(clippy::panic_in_result_fn)]
	fn run_const<E: const Expr + ExprToDyn>(
		&self,
		output: &Tensor,
		elem_args: [&Tensor; E::ELEMWISE_COUNT],
		reduce_args: [&Tensor; E::REDUCE_COUNT],
		scalar_args: [f64; E::SCALAR_COUNT],
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); E::PADDED_KEY_LEN]:,
		[(); E::BATCHED_KEY_LEN]:,
		[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:,
	{
		let (key, key_hash) = const { Self::const_key::<E>() };
		let cache = unsafe { &mut *self.cache.get() };
		let entry = if let Some(entry) = cache.find(&key, key_hash) {
			entry
		} else {
			cold_path();
			let (mut e, mut r, mut s) = (0, 0, 0);
			let dyn_expr = E::to_dyn(&mut e, &mut r, &mut s, false);
			assert!(e == E::ELEMWISE_COUNT);
			assert!(r == E::REDUCE_COUNT);
			assert!(s == E::SCALAR_COUNT);

			let mut registry = self.registry.write().unwrap();
			let kernel = registry.add_kernel(&key, key_hash, |id| {
				Arc::new(KernelData {
					id,
					key: key.into(),
					expr: dyn_expr,
					elemwise_count: E::ELEMWISE_COUNT,
					reduce_count: E::REDUCE_COUNT,
					scalar_count: E::SCALAR_COUNT,
				})
			});
			std::mem::drop(registry);

			cache.insert_unique(key_hash, kernel)
		};
		let kernel = entry.value.as_ref();
		Self::dispatch(kernel, output, elem_args, reduce_args, scalar_args)
	}

	#[allow(clippy::indexing_slicing)]
	const fn const_key<E: const Expr>() -> ([u64; E::BATCHED_KEY_LEN], u64)
	where
		[(); E::PADDED_KEY_LEN]:,
	{
		let mut discriminants = [ExprDiscriminant::Invalid; E::PADDED_KEY_LEN];
		let len = E::key(&mut discriminants, 0);
		assert!(len == E::KEY_LEN);

		let key_union = KeyUnion { discriminants };
		let key = unsafe { key_union.key };
		let key_hash = KernelMap::hash_key(&key);

		(key, key_hash)
	}

	#[inline(never)]
	#[allow(clippy::indexing_slicing)]
	#[allow(clippy::too_many_lines)]
	fn dispatch<const E: usize, const R: usize, const C: usize>(
		kernel_data: &KernelData,
		output: &Tensor,
		elem_args: [&Tensor; E],
		reduce_args: [&Tensor; R],
		scalar_args: [f64; C],
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); 1 + E + R]:,
	{
		debug_assert!(kernel_data.elemwise_count == E);
		debug_assert!(kernel_data.reduce_count == R);
		debug_assert!(kernel_data.scalar_count == C);

		let dtype_bytes = output.buf().dtype().bytes();
		debug_assert!(dtype_bytes > 0);

		let output_batch_dims: &[SizeAndStride];
		let elem_args_batch_dims: [&[SizeAndStride]; E];
		let reduce_args_batch_dims: [&[SizeAndStride]; R];
		let reduce_args_top_dim: [SizeAndStride; R];
		if R == 0 {
			reduce_args_top_dim = [SizeAndStride::default(); R];
			reduce_args_batch_dims = [&[]; R];

			output_batch_dims = output.map().dims.as_slice();
			elem_args_batch_dims = elem_args.map(|t| t.map().dims.as_slice());
		} else {
			let output_dims = output.map().dims.as_slice().split_last();
			let elem_args_dims = elem_args.try_map(|t| t.map().dims.as_slice().split_last());
			let reduce_args_dims = reduce_args.try_map(|t| t.map().dims.as_slice().split_last());

			let (Some(output_dims), Some(elem_dims), Some(reduce_dims)) =
				(output_dims, elem_args_dims, reduce_args_dims)
			else {
				cold_path();
				return Err(TensorOpError::missing_reduce_dimension());
			};

			let output_top_dim = output_dims.0;
			output_batch_dims = output_dims.1;
			let elem_args_top_dim = elem_dims.map(|(&top_dim, _)| top_dim);
			elem_args_batch_dims = elem_dims.map(|(_, batch_dim)| batch_dim);
			reduce_args_top_dim = reduce_dims.map(|(&top_dim, _)| top_dim);
			reduce_args_batch_dims = reduce_dims.map(|(_, batch_dim)| batch_dim);

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
		}

		let all_dims_tmp =
			crate::util::array::concat_arrays([output_batch_dims], elem_args_batch_dims);
		let all_dims = crate::util::array::concat_arrays(all_dims_tmp, reduce_args_batch_dims);

		let merged = DimMerger::merge::<2>(all_dims)?;

		let reduce_args_top_dim = DimMerger::merge_single_dim(reduce_args_top_dim)?;
		let reduce_inp: [KernelReduceArg; R] = std::array::from_fn(|i| {
			let arg = reduce_args[i];
			KernelReduceArg {
				stride_bytes: [
					merged[0].strides[1 + E + i] * dtype_bytes,
					merged[1].strides[1 + E + i] * dtype_bytes,
					reduce_args_top_dim.strides[i] * dtype_bytes,
				],
				offset_bytes: arg.map().offset * dtype_bytes,
				device_data: arg.buf().device_data(),
			}
		});

		let inp: [KernelElemArg; E] = std::array::from_fn(|i| {
			let arg = elem_args[i];
			KernelElemArg {
				stride_bytes: [
					merged[0].strides[1 + i] * dtype_bytes,
					merged[1].strides[1 + i] * dtype_bytes,
				],
				offset_bytes: arg.map().offset * dtype_bytes,
				device_data: arg.buf().device_data(),
			}
		});

		let out = [KernelOutput {
			size: [merged[0].size, merged[1].size],
			stride_bytes: [
				merged[0].strides[0] * dtype_bytes, //
				merged[1].strides[0] * dtype_bytes,
			],
			offset_bytes: output.map().offset * dtype_bytes,
			device_data: output.buf().device_data(),
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
					&& likely(
						inp[i].offset_bytes == out[0].offset_bytes
							&& inp[i].stride_bytes == out[0].stride_bytes,
					);
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

			output.vmt().run_kernel(
				kernel_data,
				out.as_ptr(),
				inp.as_ptr(),
				reduce_inp.as_ptr(),
				scalar_args.as_ptr(),
				reduce_args_top_dim.size,
			)?;

			std::mem::drop(out_borrow);
			std::mem::drop(elem_borrows);
			std::mem::drop(reduce_borrows);
		}
		Ok(())
	}
}
