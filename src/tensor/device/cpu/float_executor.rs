//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::{Cell, RefCell};
use std::hint::cold_path;
use std::rc::Rc;

use crate::tensor::device::cpu::rng::Rng;
use crate::tensor::device::executor::{Executor, SliceBatch};
use crate::tensor::generic::map::CompactND;
use crate::tensor::{HasDType, generic};
use crate::{Error, Result, s};

pub type CPUSliceBatch<'a, T> = generic::Tensor<CompactND<2>, &'a [Cell<T>]>;

pub struct FloatExecutor<T: HasDType> {
	rng: Rc<RefCell<Rng>>,
	phantom: std::marker::PhantomData<T>,
}

impl<T: HasDType> FloatExecutor<T> {
	pub fn new(rng: Rc<RefCell<Rng>>) -> Self {
		Self { rng, phantom: std::marker::PhantomData }
	}

	/// # Errors
	/// - If all the batches don't have the same size.
	/// - If any of the inputs don't have a safe map.
	pub fn slice_wise<const N: usize>(
		batch: [CPUSliceBatch<T>; N], mut f: impl FnMut([&[Cell<T>]; N]),
	) -> Result<()> {
		for item in batch {
			if !item.is_map_safe() {
				#[cold]
				fn err_slice_batch_not_safe() -> Error {
					"Unsafe SliceBatch detected".into()
				}
				return Err(err_slice_batch_not_safe());
			}
		}

		let count = batch.first().map_or(0, |s| s.map.shape[0]);
		if batch.iter().any(|s| s.map.shape[0] != count) {
			#[cold]
			fn err_batch_sizes_not_equal() -> Error {
				"Batches must have the same size".into()
			}
			return Err(err_batch_sizes_not_equal());
		}

		for i in 0..count {
			let slices = batch.map(|s| {
				unsafe {
					// SAFETY: `i < count` && we verified that all batches have size `count`
					let s = s.select_unchecked(s![i, ..]);
					// SAFETY:
					// - `SliceBatch` uses `CompactND` as `Map`, which guarantees that the selected
					//   sub-tensor is contiguous
					// - At the start of the function, we checked that all slices have safe maps.
					s.as_slice_unchecked()
				}
			});
			f(slices);
		}

		Ok(())
	}

	/// # Errors
	/// - If all the batches don't have the same shape.
	/// - If any of the inputs doesn't have a safe map.
	pub fn elem_wise<const N: usize>(
		batch: [CPUSliceBatch<T>; N], mut f: impl FnMut([&Cell<T>; N]),
	) -> Result<()> {
		let slice_len = batch.first().map_or(0, |&s| s.map.shape[1]);
		if batch.iter().any(|i| i.map.shape[1] != slice_len) {
			cold_path();
			return Err("Batches must have the same slice length".into());
		}

		Self::slice_wise(batch, |slices| {
			for i in 0..slice_len {
				f(slices.map(|slice| {
					debug_assert!(i < slice.len());
					let element = unsafe { slice.get_unchecked(i) };
					element
				}));
			}
		})
	}

	/// # Errors
	/// - If 'dst' doesn't have a safe map.
	pub fn nullary(dst: CPUSliceBatch<T>, mut f: impl FnMut(&Cell<T>)) -> Result<()> {
		Self::slice_wise([dst], |[dst]| {
			for d in dst {
				f(d);
			}
		})
	}

	/// # Errors
	/// - If both tensors don't have the same shape.
	/// - If any of the tensors doesn't have a safe map.
	pub fn unary(
		dst: CPUSliceBatch<T>, inp: CPUSliceBatch<T>, mut f: impl FnMut(&Cell<T>, &Cell<T>),
	) -> Result<()> {
		if inp.map.shape[0] != dst.map.shape[0]
			|| (inp.map.shape[1] != dst.map.shape[1] && inp.map.shape[1] != 1)
		{
			#[cold]
			fn err_batches_shape_mismatch() -> Error {
				"Batches must have the same shape".into()
			}
			return Err(err_batches_shape_mismatch());
		}

		if inp.map.shape[1] == 1 {
			Self::slice_wise([dst, inp], |[dst, inp]| {
				debug_assert!(inp.len() == 1);
				// SAFETY: `inp.map.shape[1] == 1`, so `inp` has only one element and so we can
				// safely use `first()`.
				let i = unsafe { inp.first().unwrap_unchecked() };
				for d in dst {
					f(d, i);
				}
			})
		} else {
			Self::slice_wise([dst, inp], |[dst, inp]| {
				for (d, i) in dst.iter().zip(inp.iter()) {
					f(d, i);
				}
			})
		}
	}
}

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

impl<T: HasDType + Copy + FromToF64> Executor for FloatExecutor<T> {
	fn zeros(&self, dst: &SliceBatch) -> Result<()> {
		let dst = dst.try_view()?;
		Self::nullary(dst, |d| d.set(T::from_f64(0.0)))
	}

	fn randn_clamped(&self, dst: &SliceBatch) -> Result<()> {
		let dst = dst.try_view()?;
		let mut rng = self.rng.borrow_mut();
		Self::nullary(dst, |d| d.set(T::from_f64(rng.get_normal_clamped())))
	}

	fn copy(&self, dst: &SliceBatch, src: &SliceBatch) -> Result<()> {
		let dst = dst.try_view()?;
		let src = src.try_view()?;
		Self::unary(dst, src, |d, s| d.set(s.get()))
	}

	fn acc(
		&self, dst: &SliceBatch, dst_weight: f64, upd: &SliceBatch, upd_weight: f64,
	) -> Result<()> {
		let dst = dst.try_view()?;
		let upd = upd.try_view()?;
		Self::unary(dst, upd, |d, u| {
			let d_val = d.get().to_f64();
			let u_val = u.get().to_f64();

			// Justification for allowing suboptimal_flops:
			// Clippy recommends using `mul_add()`, however I checked the assembly and
			// it generates `callq	*fma@GOTPCREL(%rip)`, which will probably be incredibly slow.
			#[allow(clippy::suboptimal_flops)]
			let v = (d_val * dst_weight) + (u_val * upd_weight);

			d.set(T::from_f64(v));
		})
	}

	fn mul(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()> {
		// TODO - this doesn't handle broadcasting
		let dst = dst.try_view()?;
		let a = a.try_view()?;
		let b = b.try_view()?;
		Self::elem_wise([dst, a, b], |[d, a, b]| {
			let v = a.get().to_f64() * b.get().to_f64();
			d.set(T::from_f64(v));
		})
	}
}
