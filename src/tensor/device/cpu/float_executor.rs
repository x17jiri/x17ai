//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;

use crate::s;
use crate::tensor::device::executor::{Executor, SliceBatch};
use crate::tensor::generic::map::CompactND;
use crate::tensor::{HasDType, generic};

pub type CPUSliceBatch<'a, T> = generic::Tensor<CompactND<2>, &'a [Cell<T>]>;

pub struct FloatExecutor<T: HasDType> {
	phantom: std::marker::PhantomData<T>,
}

impl<T: HasDType> FloatExecutor<T> {
	pub fn new() -> Self {
		FloatExecutor { phantom: std::marker::PhantomData }
	}

	fn slice_wise<const N: usize>(batch: [&SliceBatch; N], mut f: impl FnMut([&[Cell<T>]; N])) {
		let batch = batch.map(|s| s.try_view::<T>().unwrap());

		let count = batch.get(0).map_or(0, |s| s.map.shape[0]);
		assert!(batch.iter().all(|s| s.map.shape[0] == count));

		for i in 0..count {
			let slices = batch.map(|s| s.slice(s![i, ..]));
			f(slices);
		}
	}

	fn elem_wise<const N: usize>(batch: [&SliceBatch; N], mut f: impl FnMut([&Cell<T>; N])) {
		let slice_len = batch.get(0).map_or(0, |&s| s.map.shape[1]);
		assert!(batch.iter().all(|i| i.map.shape[1] == slice_len));

		Self::slice_wise(batch, |slices| {
			for i in 0..slice_len {
				f(slices.map(|slice| {
					debug_assert!(i < slice.len());
					let element = unsafe { slice.get_unchecked(i) };
					element
				}));
			}
		});
	}
}

impl Executor for FloatExecutor<f32> {
	fn zeros(&self, dst: &SliceBatch) {
		Self::elem_wise([dst], |[d]| d.set(0.0))
	}
}
