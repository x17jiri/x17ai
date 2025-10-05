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
use crate::tensor::device::dtype::{self, DTypeId};
use crate::tensor::device::kernel::expr::{
	self, DynExpr, ExprToDyn, ExprTrait, KEY_TYPE_SIZE, KernelKeyType,
};
use crate::tensor::device::kernel::registry::{KernelMap, KernelRegistry};
use crate::tensor::device::{DeviceBuffer, KernelElemArg, KernelOutput, KernelReduceArg};
use crate::tensor::dim_merger::DimMerger;
use crate::tensor::generic::map::SizeAndStride;
use crate::tensor::{DType, NotContiguousError, Tensor, TensorOpError};
use crate::util::mycell::{BorrowGuard, UnsafeBorrowFailFlag, UnsafeBorrowMutFailFlag};

//--------------------------------------------------------------------------------------------------

pub struct KernelRunner {
	//	registry: Arc<RwLock<KernelRegistry>>,
	//	cache: UnsafeCell<KernelMap>,
}

impl Default for KernelRunner {
	fn default() -> Self {
		Self::new()
	}
}

impl KernelRunner {
	pub fn new() -> Self {
		Self {
//			registry: KernelRegistry::instance(),
//			cache: UnsafeCell::new(KernelMap::new()),
		}
	}

	#[inline(never)]
	#[allow(clippy::panic_in_result_fn)]
	pub fn run<E: const ExprTrait + ExprToDyn>(
		&self,
		expr: E,
		output: &Tensor,
		elem_args: [&Tensor; E::ELEMWISE_COUNT],
		reduce_args: [&Tensor; E::REDUCE_COUNT],
		scalar_args: [f64; E::SCALAR_COUNT],
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); E::KEY_WORDS]:,
		[(); KEY_TYPE_SIZE * E::KEY_WORDS]:,
		[(); KernelData::dtype_config_words(E::ELEMWISE_COUNT, E::REDUCE_COUNT)]:,
		[(); E::DTYPE_CONFIG_WORDS
			- KernelData::dtype_config_words(E::ELEMWISE_COUNT, E::REDUCE_COUNT)]:,
	{
		let mut key = const { E::key() };
		let dtype_config =
			KernelData::new_dtype_config(internal_dtype, output, elem_args, reduce_args);
		key[..dtype_config.len()].copy_from_slice(&dtype_config);

		let kernel_data = KernelData {
			elemwise_count: E::ELEMWISE_COUNT,
			reduce_count: E::REDUCE_COUNT,
			scalar_count: E::SCALAR_COUNT,
			expr: &expr,
			key: &key,
		};

		Ok(())
		/*
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
		*/
	}
}
