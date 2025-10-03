//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::UnsafeCell;
use std::hint::{cold_path, likely};
use std::sync::{Arc, RwLock};

use crate::ErrPack;
use crate::tensor::device::dtype::DTypeId;
use crate::tensor::device::kernel::expr::{DynExpr, ExprToDyn, ExprTrait};
use crate::tensor::device::kernel::registry::{KernelMap, KernelRegistry};
use crate::tensor::device::{DeviceBuffer, KernelElemArg, KernelOutput, KernelReduceArg};
use crate::tensor::dim_merger::DimMerger;
use crate::tensor::generic::map::SizeAndStride;
use crate::tensor::{DType, NotContiguousError, Tensor, TensorOpError};
use crate::util::mycell::{BorrowGuard, UnsafeBorrowFailFlag, UnsafeBorrowMutFailFlag};

//--------------------------------------------------------------------------------------------------

pub struct KernelData {
	pub id: usize,
	pub key: Box<[u64]>,
	pub expr: Arc<DynExpr>,
	pub elemwise_count: usize,
	pub reduce_count: usize,
	pub scalar_count: usize,
}

impl KernelData {
	pub fn new_dtype_config<const E: usize, const R: usize>(
		internal_dtype: DType,
		output: &Tensor,
		elem_args: [&Tensor; E],
		reduce_args: [&Tensor; R],
	) -> [u64; ((2 + E + R) * std::mem::size_of::<DTypeId>() + 7) / 8] {
		let mut result = [0; ((2 + E + R) * std::mem::size_of::<DTypeId>() + 7) / 8];
		let ptr = result.as_mut_ptr().cast::<DTypeId>();
		unsafe {
			*ptr = internal_dtype.id();
			*ptr.add(1) = output.dtype().id();
			for i in 0..E {
				*ptr.add(2 + i) = elem_args[i].dtype().id();
			}
			for i in 0..R {
				*ptr.add(2 + E + i) = reduce_args[i].dtype().id();
			}
		}
		result
	}

	pub unsafe fn elemwise_args<'a>(&self, args: *const KernelElemArg) -> &'a [KernelElemArg] {
		unsafe { std::slice::from_raw_parts(args, self.elemwise_count) }
	}
	pub unsafe fn reduce_args<'a>(&self, args: *const KernelReduceArg) -> &'a [KernelReduceArg] {
		unsafe { std::slice::from_raw_parts(args, self.reduce_count) }
	}
	pub unsafe fn scalar_args<'a>(&self, args: *const f64) -> &'a [f64] {
		unsafe { std::slice::from_raw_parts(args, self.scalar_count) }
	}

	pub fn elemwise_dtype<'a>(&self, dtype_config: &[u64], i: usize) -> DType {
		assert!(dtype_config.len() == self.dtype_config_len());
		assert!(i < self.elemwise_count);
		let dtype_config = dtype_config as *const [u64];
		let dtype_config = dtype_config.cast::<u64>().cast::<DTypeId>();
		unsafe { *dtype_config.add(2 + i) }.to_dtype()
	}
	pub fn reduce_dtype<'a>(&self, dtype_config: &[u64], i: usize) -> DType {
		assert!(dtype_config.len() == self.dtype_config_len());
		assert!(i < self.reduce_count);
		let dtype_config = dtype_config as *const [u64];
		let dtype_config = dtype_config.cast::<u64>().cast::<DTypeId>();
		unsafe { *dtype_config.add(2 + self.elemwise_count + i) }.to_dtype()
	}

	pub fn internal_dtype(&self, dtype_config: &[u64]) -> DType {
		assert!(dtype_config.len() == self.dtype_config_len());
		let dtype_config = dtype_config as *const [u64];
		let dtype_config = dtype_config.cast::<u64>().cast::<DTypeId>();
		unsafe { *dtype_config }.to_dtype()
	}
	pub fn output_dtype(&self, dtype_config: &[u64]) -> DType {
		assert!(dtype_config.len() == self.dtype_config_len());
		let dtype_config = dtype_config as *const [u64];
		let dtype_config = dtype_config.cast::<u64>().cast::<DTypeId>();
		unsafe { *dtype_config.add(1) }.to_dtype()
	}

	pub fn dtype_config_len(&self) -> usize {
		let bytes = (2 + self.elemwise_count + self.reduce_count) * std::mem::size_of::<DTypeId>();
		let items = (bytes + 7) / 8;
		items
	}

	pub unsafe fn dtype_config<'a>(&self, dtype_config: *const u64) -> &'a [u64] {
		let items = self.dtype_config_len();
		unsafe { std::slice::from_raw_parts(dtype_config, items) }
	}
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

	#[inline(never)]
	#[allow(clippy::panic_in_result_fn)]
	pub fn run<E: const ExprTrait + ExprToDyn>(
		&self,
		output: &Tensor,
		elem_args: [&Tensor; E::ELEMWISE_COUNT],
		reduce_args: [&Tensor; E::REDUCE_COUNT],
		scalar_args: [f64; E::SCALAR_COUNT],
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); E::PADDED_KEY_LEN]:,
		[(); E::BATCHED_KEY_LEN]:,
		[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:,
		[(); ((2 + E::ELEMWISE_COUNT + E::REDUCE_COUNT) * std::mem::size_of::<DTypeId>() + 7) / 8]:,
	{
		let (key, key_hash) = const { E::key() };
		let cache = unsafe { &mut *self.cache.get() };
		let entry = if let Some(entry) = cache.find(&key, key_hash) {
			entry
		} else {
			cold_path();
			let dyn_expr = E::to_dyn(E::ELEMWISE_MASK, E::REDUCE_MASK, E::SCALAR_MASK, false);

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
		Self::__run(kernel, output, elem_args, reduce_args, scalar_args, internal_dtype)
	}

	#[inline(never)]
	#[allow(clippy::indexing_slicing)]
	#[allow(clippy::too_many_lines)]
	fn __run<const E: usize, const R: usize, const C: usize>(
		kernel_data: &KernelData,
		output: &Tensor,
		elem_args: [&Tensor; E],
		reduce_args: [&Tensor; R],
		scalar_args: [f64; C],
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); 1 + E + R]:,
		[(); ((2 + E + R) * std::mem::size_of::<DTypeId>() + 7) / 8]:,
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
				return Err(TensorOpError::cannot_broadcast_output());
			}
			if reduce_args_top_dim.iter().any(|vec| vec.stride != 1) {
				cold_path();
				return Err(NotContiguousError.into());
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
				buf: arg.buf().memory(),
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
				buf: arg.buf().memory(),
			}
		});

		if merged.iter().any(|m| m.get(0).is_broadcasted()) {
			cold_path();
			return Err(TensorOpError::cannot_broadcast_output());
		}

		let out = KernelOutput {
			size: [merged[0].size, merged[1].size],
			stride_bytes: [
				merged[0].strides[0] * dtype_bytes, //
				merged[1].strides[0] * dtype_bytes,
			],
			offset_bytes: output.map().offset * dtype_bytes,
			buf: output.buf().memory(),
			reduction_size: reduce_args_top_dim.size,
		};

		let dtype_config =
			KernelData::new_dtype_config(internal_dtype, output, elem_args, reduce_args);

		unsafe {
			let mut inp_fail = UnsafeBorrowFailFlag::new();
			let reduce_borrows: [BorrowGuard<DeviceBuffer>; R] =
				std::array::from_fn(|i| reduce_args[i].buf().unsafe_borrow(&mut inp_fail));
			let elem_borrows: [Option<BorrowGuard<DeviceBuffer>>; E] = std::array::from_fn(|i| {
				let arg = &elem_args[i];
				let same_as_output = std::ptr::eq(arg.buf().as_ref(), output.buf().as_ref())
					&& likely(
						inp[i].offset_bytes == out.offset_bytes
							&& inp[i].stride_bytes == out.stride_bytes,
					);
				if same_as_output { None } else { Some(arg.buf().unsafe_borrow(&mut inp_fail)) }
			});

			let mut out_fail = UnsafeBorrowMutFailFlag::new();
			let out_borrow = output.buf().unsafe_borrow_mut(&mut out_fail);

			inp_fail.check()?;
			out_fail.check()?;

			// TODO - ensure_safe
			// TODO - ensure all on same device
			// TODO - other things may need to be checked before running the kernel

			output.device().run_kernel(
				kernel_data,
				&out,
				inp.as_ptr(),
				reduce_inp.as_ptr(),
				scalar_args.as_ptr(),
				dtype_config.as_ptr(),
			)?;

			std::mem::drop(out_borrow);
			std::mem::drop(elem_borrows);
			std::mem::drop(reduce_borrows);
		}
		Ok(())
	}
}
