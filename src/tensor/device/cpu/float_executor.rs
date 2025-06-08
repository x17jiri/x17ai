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
use crate::tensor::device::cpu::zip::{SliceMap, reduce_zip_n, vec_zip_n};
use crate::tensor::device::executor::{Executor, SliceBatch, ensure_same_shape};
use crate::util::LossyInto;
use crate::util::array::{concat_arrays, try_map_borrowed};

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
		ensure_same_shape([dst, inp])?;
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
		ensure_same_shape([dst, a, b])?;
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
		ensure_same_shape(t)?;
		let t = try_map_borrowed(&t, |_, t| CPUInput::new_safe_contiguous(t))?;
		unsafe {
			zip_n(t, f);
		}
		Ok(())
	}

	pub fn n_vec<const N: usize>(
		t: [&SliceBatch; N], f: impl FnMut([&[Cell<T>]; N]),
	) -> Result<()> {
		ensure_same_shape(t)?;
		let t = try_map_borrowed(&t, |_, t| CPUInput::new_safe_contiguous(t))?;
		unsafe {
			vec_zip_n(t, f);
		}
		Ok(())
	}

	pub fn vec_reduce<const N: usize>(
		r: &SliceBatch, a: [&SliceBatch; N], f: impl FnMut(&Cell<T>, [&[Cell<T>]; N]),
	) -> Result<()>
	where
		[(); 1 + N]:,
	{
		let a_shape = ensure_same_shape(a)?;
		let r_shape = r.nd_shape()?;
		if r_shape[0] != a_shape[0] || r_shape[1] != 1 {
			#[cold]
			fn err_reduce_shape(r_shape: [usize; 2], a_shape: [usize; 2]) -> crate::Error {
				let r0 = r_shape[0];
				let r1 = r_shape[1];
				let a0 = a_shape[0];
				format!("Invalid output shape. Expected [{a0}, 1], got [{r0}, {r1}].",).into()
			}
			return Err(err_reduce_shape(r_shape, a_shape));
		}
		let r = CPUInput::new_safe_contiguous(r)?;
		let a = try_map_borrowed(&a, |_, a| CPUInput::new_safe_contiguous(a))?;
		unsafe {
			reduce_zip_n(r, a, f);
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
			let a = a.get().to_f64();

			sum += a;
		})?;
		Ok(sum)
	}

	fn approx_eq(&self, a: &SliceBatch, b: &SliceBatch, eps: f64) -> Result<bool> {
		// TODO - this could handle broadcasted tensors as well
		let mut result = true;
		Self::n_contiguous([a, b], |[a, b]| {
			let a = a.get().to_f64();
			let b = b.get().to_f64();

			result &= math::approx_eq(a, b, eps);
		})?;
		Ok(result)
	}

	fn softmax(&self, out: &SliceBatch, inp: &SliceBatch) -> Result<()> {
		Self::n_vec([out, inp], |[out, inp]| math::softmax(out, inp))
	}

	fn softmax_backward(
		&self, d_inp: &SliceBatch, out: &SliceBatch, d_out: &SliceBatch,
	) -> Result<()> {
		Self::n_vec([d_inp, out, d_out], |[d_inp, out, d_out]| {
			math::softmax_backward(d_inp, out, d_out);
		})
	}

	fn rms_norm(&self, out: &SliceBatch, inp: &SliceBatch, eps: f64) -> Result<()> {
		Self::n_vec([out, inp], |[out, inp]| math::rms_norm(out, inp, eps))
	}

	fn dot(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64) -> Result<()> {
		Self::vec_reduce(dst, [a, b], |dst, [a, b]| {
			let v = math::dot(a, b) * ab_weight;
			dst.set(T::from_f64(v));
		})
	}

	fn dot_acc(
		&self, dst: &SliceBatch, dst_weight: f64, a: &SliceBatch, b: &SliceBatch, ab_weight: f64,
	) -> Result<()> {
		Self::vec_reduce(dst, [a, b], |dst, [a, b]| {
			let d = dst.get().to_f64();
			let v = math::dot(a, b);
			let v = d * dst_weight + v * ab_weight;
			dst.set(T::from_f64(v));
		})
	}

	fn rsqrt_dot(
		&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64, eps: f64,
	) -> Result<()> {
		Self::vec_reduce(dst, [a, b], |dst, [a, b]| {
			let v = math::dot(a, b) * ab_weight;
			let v = math::rsqrt(v + eps);
			dst.set(T::from_f64(v));
		})
	}
}
