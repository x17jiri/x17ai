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
use crate::util::LossyInto;
use crate::util::array::{try_map_borrowed, try_map_into};
use crate::{Error, Result};

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

	pub fn ensure_same_shape<const N: usize>(t: [&SliceBatch; N]) -> Result<()> {
		let shapes = try_map_into(t, |_, t| t.nd_shape())?;
		if let Some(shape) = shapes.first()
			&& shapes.iter().any(|s| s != shape)
		{
			#[cold]
			fn err_shape_mismatch<const N: usize>(shapes: &[[usize; 2]]) -> Error {
				let shapes_str = shapes
					.iter()
					.map(|[a, b]| format!("[{a}, {b}]"))
					.collect::<Vec<_>>()
					.join(", ");
				format!("Expected all tensors to have the same shape, but got: {shapes_str}").into()
			}
			return Err(err_shape_mismatch::<N>(&shapes));
		}
		Ok(())
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
		Self::ensure_same_shape([dst, inp])?;
		let dst = CPUInput::new_safe_contiguous(dst)?;
		let inp = CPUInput::new_safe(inp)?;
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
		Self::ensure_same_shape([dst, a, b])?;
		let dst = CPUInput::new_safe_contiguous(dst)?;
		let a = CPUInput::new_safe(a)?;
		let b = CPUInput::new_safe(b)?;
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
		Self::ensure_same_shape(t)?;
		let t = try_map_borrowed(&t, |_, t| CPUInput::new_safe_contiguous(t))?;
		unsafe {
			zip_n(t, f);
		}
		Ok(())
	}

	pub fn n_vec<const N: usize>(
		t: [&SliceBatch; N], f: impl FnMut([&[Cell<T>]; N]),
	) -> Result<()> {
		Self::ensure_same_shape(t)?;
		let t = try_map_borrowed(&t, |_, t| CPUInput::new_safe_contiguous(t))?;
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

			let v = d * dst_weight + u * upd_weight;

			dst.set(T::from_f64(v));
		})
	}

	fn rsqrt(&self, dst: &SliceBatch, a: &SliceBatch, eps: f64) -> Result<()> {
		Self::unary(dst, a, |dst, a| {
			let a = a.get().to_f64();

			let v = math::rsqrt(a + eps);

			dst.set(T::from_f64(v));
		})
	}

	fn ln_clamped(&self, dst: &SliceBatch, a: &SliceBatch) -> Result<()> {
		Self::unary(dst, a, |dst, a| {
			let a = a.get().to_f64();

			let v = a.ln().max(-1000.0);

			dst.set(T::from_f64(v));
		})
	}

	fn mul(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()> {
		Self::binary(dst, a, b, |dst, a, b| {
			let a = a.get().to_f64();
			let b = b.get().to_f64();

			let v = a * b;

			dst.set(T::from_f64(v));
		})
	}

	fn mul_acc(
		&self, dst: &SliceBatch, dst_weight: f64, a: &SliceBatch, b: &SliceBatch, ab_weight: f64,
	) -> Result<()> {
		Self::binary(dst, a, b, |dst, a, b| {
			let d = dst.get().to_f64();
			let a = a.get().to_f64();
			let b = b.get().to_f64();

			let v = (d * dst_weight) + (a * b * ab_weight);

			dst.set(T::from_f64(v));
		})
	}

	fn sub(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()> {
		Self::binary(dst, a, b, |dst, a, b| {
			let a = a.get().to_f64();
			let b = b.get().to_f64();

			let v = a - b;

			dst.set(T::from_f64(v));
		})
	}

	fn add(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()> {
		Self::binary(dst, a, b, |dst, a, b| {
			let a = a.get().to_f64();
			let b = b.get().to_f64();

			let v = a + b;

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

	fn sum_all(&self, a: &SliceBatch) -> Result<f64> {
		// TODO - this could handle broadcasted tensors as well
		let mut sum = 0.0;
		Self::n_contiguous([a], |[a]| {
			sum += a.get().to_f64();
		})?;
		Ok(sum)
	}
	fn approx_eq(&self, a: &SliceBatch, b: &SliceBatch, eps: f64) -> Result<bool> {
		// TODO - this could handle broadcasted tensors as well
		let mut result = true;
		Self::n_contiguous([a, b], |[a, b]| {
			result &= math::approx_eq(a.get().to_f64(), b.get().to_f64(), eps);
		})?;
		Ok(result)
	}

	fn softmax(&self, dst: &SliceBatch, a: &SliceBatch) -> Result<()> {
		Self::n_vec([dst, a], |[dst, a]| math::softmax(dst, a))
	}

	fn rms_norm(&self, dst: &SliceBatch, a: &SliceBatch, eps: f64) -> Result<()> {
		Self::n_vec([dst, a], |[dst, a]| {
			//let len = dst.map.dims[1].size;
			let len = dst.len();
			let len: f64 = len.lossy_into();
			let len_recip = 1.0 / len;

			//--

			let scale = math::rsqrt(math::dot(a, a) * len_recip + eps);
			for (d, i) in dst.iter().zip(a) {
				let val = i.get().to_f64() * scale;
				d.set(T::from_f64(val));
			}
		})
	}

	/*
	fn dot(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64) -> Result<()> {
		// TODO - ensure shape
		Self::n_vec([dst, a, b], |[dst, a, b]| {
			let val = math::dot(a, b) * ab_weight;
			let val = T::from_f64(val);
			dst[0].set(val);
		})
	}
	*/
}
