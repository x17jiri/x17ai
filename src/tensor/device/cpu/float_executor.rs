//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::Result;
use crate::tensor::HasDType;
use crate::tensor::device::executor::{Executor, SliceBatch};

use super::rng::Rng;
use super::zip::{CPUInput, zip1, zip2, zip3};

//--------------------------------------------------------------------------------------------------

pub trait FromToF64 {
	const MIN: f64; // largest negative value of type

	fn from_f64(val: f64) -> Self;
	fn to_f64(&self) -> f64;
}

#[allow(clippy::use_self)]
impl FromToF64 for f32 {
	const MIN: f64 = f32::MIN as f64;

	fn from_f64(val: f64) -> Self {
		#[allow(clippy::cast_possible_truncation)]
		(val as f32)
	}

	fn to_f64(&self) -> f64 {
		f64::from(*self)
	}
}

#[allow(clippy::use_self)]
impl FromToF64 for f64 {
	const MIN: f64 = f64::MIN;

	fn from_f64(val: f64) -> Self {
		val
	}

	fn to_f64(&self) -> f64 {
		*self
	}
}

//--------------------------------------------------------------------------------------------------

pub struct FloatExecutor<T: Copy + HasDType> {
	rng: Rc<RefCell<Rng>>,
	phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + HasDType> FloatExecutor<T> {
	pub fn new(rng: Rc<RefCell<Rng>>) -> Self {
		Self { rng, phantom: std::marker::PhantomData }
	}

	/*
		/// # Safety
		/// - batch_size - All inputs must have equal shape[0], i.e., the batch size.
		/// - safe_map - All inputs must have safe maps.
		pub unsafe fn slice_wise<const N: usize>(
			batch: [CPUSliceBatch<T>; N], mut f: impl FnMut([&[Cell<T>]; N]),
		) {
			let count = batch.first().map_or(0, |s| s.map.shape[0]);
			for i in 0..count {
				f(batch.map(|s| {
					unsafe {
						// SAFETY:
						// - dimensionality - `CPUSliceBatch` has 2 dimensions
						// - valid_ranges - The index `i` is valid if all inputs have the same shape[0],
						// which is one of the preconditions of this function.
						let s = s.select_unchecked(s![i, ..]);
						// SAFETY:
						// - contiguous - `CPUSliceBatch` uses `CompactND` as `Map`, which guarantees
						//   that the selected sub-tensor is contiguous
						// - safe_map - This is a precondition of this function.
						s.as_slice_unchecked()
					}
				}));
			}
		}
	*/
	/// # Errors
	/// - If 'dst' doesn't have a safe map.
	pub fn nullary(dst: &SliceBatch, f: impl FnMut(&Cell<T>)) -> Result<()> {
		let dst = CPUInput::new_safe_contiguous(dst)?;
		unsafe {
			zip1(dst, f);
		}
		Ok(())
	}

	/// # Errors
	/// - If both tensors don't have the same shape.
	/// - If any of the tensors doesn't have a safe map.
	pub fn unary(
		dst: &SliceBatch, inp: &SliceBatch, mut f: impl FnMut(&Cell<T>, &Cell<T>),
	) -> Result<()> {
		let dst = CPUInput::new_safe_contiguous(dst)?;
		let inp = CPUInput::new_safe(inp)?;

		// TODO - assert shape

		unsafe {
			match inp {
				CPUInput::Slice(inp) => {
					zip2(dst, inp, |d, i| f(d, i));
				},
				CPUInput::Broadcast(inp) => {
					zip2(dst, inp, |d, i| f(d, i));
				},
			}
		}
		Ok(())
	}

	pub fn binary(
		dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch,
		mut f: impl FnMut(&Cell<T>, &Cell<T>, &Cell<T>),
	) -> Result<()> {
		let dst = CPUInput::new_safe_contiguous(dst)?;
		let a = CPUInput::new_safe(a)?;
		let b = CPUInput::new_safe(b)?;

		// TODO - assert shape

		unsafe {
			match (a, b) {
				(CPUInput::Slice(a), CPUInput::Slice(b)) => {
					zip3(dst, a, b, |d, a, b| f(d, a, b));
				},
				(CPUInput::Broadcast(a), CPUInput::Broadcast(b)) => {
					zip3(dst, a, b, |d, a, b| f(d, a, b));
				},
				(CPUInput::Slice(a), CPUInput::Broadcast(b)) => {
					zip3(dst, a, b, |d, a, b| f(d, a, b));
				},
				(CPUInput::Broadcast(a), CPUInput::Slice(b)) => {
					zip3(dst, a, b, |d, a, b| f(d, a, b));
				},
			}
		}
		Ok(())
	}
}

impl<T: HasDType + Copy + FromToF64> Executor for FloatExecutor<T> {
	fn zeros(&self, dst: &SliceBatch) -> Result<()> {
		Self::nullary(dst, |dst| dst.set(T::from_f64(0.0)))
	}

	fn randn_clamped(&self, dst: &SliceBatch) -> Result<()> {
		let mut rng = self.rng.borrow_mut();
		Self::nullary(dst, |dst| dst.set(T::from_f64(rng.get_normal_clamped())))
	}

	fn copy(&self, dst: &SliceBatch, src: &SliceBatch) -> Result<()> {
		Self::unary(dst, src, |dst, src| dst.set(src.get()))
	}

	fn acc(
		&self, dst: &SliceBatch, dst_weight: f64, upd: &SliceBatch, upd_weight: f64,
	) -> Result<()> {
		Self::unary(dst, upd, |dst, upd| {
			let d = dst.get().to_f64();
			let u = upd.get().to_f64();

			// Justification for allowing suboptimal_flops:
			// Clippy recommends using `mul_add()`, however I checked the assembly and
			// it generates `callq	*fma@GOTPCREL(%rip)`, which will probably be incredibly slow.
			#[allow(clippy::suboptimal_flops)]
			let v = (d * dst_weight) + (u * upd_weight);

			dst.set(T::from_f64(v));
		})
	}

	fn mul(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()> {
		Self::binary(dst, a, b, |dst, a, b| {
			let v = a.get().to_f64() * b.get().to_f64();
			dst.set(T::from_f64(v));
		})
	}

	fn mul_acc(
		&self, dst: &SliceBatch, dst_weight: f64, a: &SliceBatch, b: &SliceBatch, ab_weight: f64,
	) -> Result<()> {
		Self::binary(dst, a, b, |dst, a, b| {
			let d_val = dst.get().to_f64();
			let a_val = a.get().to_f64();
			let b_val = b.get().to_f64();

			// Justification for allowing suboptimal_flops:
			// Clippy recommends using `mul_add()`, however I checked the assembly and
			// it generates `callq	*fma@GOTPCREL(%rip)`, which will probably be incredibly slow.
			#[allow(clippy::suboptimal_flops)]
			let v = (d_val * dst_weight) + (a_val * b_val * ab_weight);

			dst.set(T::from_f64(v));
		})
	}

	fn sub(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()> {
		Self::binary(dst, a, b, |dst, a, b| {
			let v = a.get().to_f64() - b.get().to_f64();
			dst.set(T::from_f64(v));
		})
	}

	fn add(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()> {
		Self::binary(dst, a, b, |dst, a, b| {
			let v = a.get().to_f64() + b.get().to_f64();
			dst.set(T::from_f64(v));
		})
	}
}
