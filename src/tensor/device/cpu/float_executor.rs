//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::tensor::HasDType;
use crate::tensor::device::cpu::zip::vec_zip_n;
use crate::tensor::device::executor::{Executor, SliceBatch};
use crate::{Result, util};

use super::math::{self, FromToF64};
use super::rng::Rng;
use super::zip::{CPUInput, zip_n, zip1, zip2, zip3};

//--------------------------------------------------------------------------------------------------

pub struct FloatExecutor<T: Copy + HasDType> {
	rng: Rc<RefCell<Rng>>,
	phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + HasDType> FloatExecutor<T> {
	pub fn new(rng: Rc<RefCell<Rng>>) -> Self {
		Self { rng, phantom: std::marker::PhantomData }
	}

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

	pub fn n_contiguous<const N: usize>(
		t: [&SliceBatch; N], f: impl FnMut([&Cell<T>; N]),
	) -> Result<()> {
		let t = util::array::try_map_borrowed(&t, |_, t| CPUInput::new_safe_contiguous(t))?;

		// TODO - assert shape

		unsafe {
			zip_n(t, f);
		}
		Ok(())
	}

	pub fn n_vec<const N: usize>(
		t: [&SliceBatch; N], f: impl FnMut([&[Cell<T>]; N]),
	) -> Result<()> {
		let t = util::array::try_map_borrowed(&t, |_, t| CPUInput::new_safe_contiguous(t))?;

		// TODO - assert shape

		unsafe {
			vec_zip_n(t, f);
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

	fn swiglu(&self, dst: &SliceBatch, lin: &SliceBatch, gate: &SliceBatch) -> Result<()> {
		Self::n_contiguous([dst, lin, gate], |[dst, lin, gate]| {
			let forward = math::swiglu(lin.get().to_f64(), gate.get().to_f64());
			dst.set(T::from_f64(forward));
		})
	}

	fn swiglu_backward(
		&self, d_lin: &SliceBatch, d_gate: &SliceBatch, lin: &SliceBatch, gate: &SliceBatch,
		d_out: &SliceBatch,
	) -> Result<()> {
		Self::n_contiguous(
			[d_lin, d_gate, lin, gate, d_out],
			|[d_lin, d_gate, lin, gate, d_out]| {
				let lin = lin.get().to_f64();
				let gate = gate.get().to_f64();
				let d_out = d_out.get().to_f64();
				let (d_lin_val, d_gate_val) = math::swiglu_backward(lin, gate);
				let d_lin_val = d_lin_val * d_out;
				let d_gate_val = d_gate_val * d_out;
				d_lin.set(T::from_f64(d_lin_val));
				d_gate.set(T::from_f64(d_gate_val));
			},
		)
	}

	fn dot(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64) -> Result<()> {
		// TODO - ensure shape
		Self::n_vec([dst, a, b], |[dst, a, b]| {
			let val = math::dot(a, b) * ab_weight;
			let val = T::from_f64(val);
			dst[0].set(val);
		})
	}
}
