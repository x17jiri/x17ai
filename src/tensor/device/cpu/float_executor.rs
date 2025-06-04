//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::Error;
use crate::tensor::device::cpu::CPUSliceSet;
use crate::tensor::device::executor::{Executor, SliceBatch};
use crate::tensor::generic::map::CompactND;
use crate::tensor::{HasDType, generic};

pub type CPUSliceBatch<'a, T> = generic::Tensor<CompactND<2>, &'a [T]>;

impl<'a, T: HasDType> TryFrom<&'a SliceBatch<'a>> for CPUSliceBatch<'a, T> {
	type Error = Error;

	fn try_from(value: &SliceBatch<'a>) -> Result<Self, Self::Error> {
		let map = value.map;
	}
}

pub struct FloatExecutor<T> {
	phantom: std::marker::PhantomData<T>,
}

impl<T> FloatExecutor<T> {
	pub fn new() -> Self {
		FloatExecutor { phantom: std::marker::PhantomData }
	}
}

impl Executor for FloatExecutor<f32> {
	fn zeros(&self, dst: &SliceBatch) {
		let dst = dst.try_view::<f32>().unwrap();
	}
}
