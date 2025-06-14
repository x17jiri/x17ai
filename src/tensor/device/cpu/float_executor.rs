//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::tensor::HasDType;
use crate::tensor::device::cpu::zip::{reduce_zip_n, vec_zip_n, zip};
use crate::tensor::device::executor::{Executor, MatrixBatch, SliceBatch, ensure_same_shape};
use crate::util::array;
use crate::{Error, Result};

use super::math::{self, FromToF64};
use super::rng::Rng;
use super::zip::{CPUInput, zip_n, zip1, zip2, zip3};

//--------------------------------------------------------------------------------------------------

#[cold]
fn err_tensor_has_stride() -> Error {
	"Tensor data is neither contiguous nor broadcasted.".into()
}

#[cold]
fn err_tensor_not_contiguous() -> Error {
	"Tensor data is not contiguous.".into()
}

pub struct FloatExecutor<T: Copy + HasDType + FromToF64> {
	rng: Rc<RefCell<Rng>>,
	phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + HasDType + FromToF64> FloatExecutor<T> {
	pub fn new(rng: Rc<RefCell<Rng>>) -> Self {
		Self { rng, phantom: std::marker::PhantomData }
	}

	pub fn ensure_safe_contiguous(tensor: &SliceBatch) -> Result<()> {
		tensor.ensure_safe()?;
		let dim = tensor.map.dims[1];
		if !dim.is_contiguous() {
			return Err(err_tensor_not_contiguous());
		}
		Ok(())
	}

	pub fn ensure_safe_contiguous_or_broadcasted(tensor: &SliceBatch) -> Result<bool> {
		tensor.ensure_safe()?;
		let dim = tensor.map.dims[1];
		if dim.is_contiguous() {
			Ok(false)
		} else if dim.is_broadcasted() {
			Ok(true)
		} else {
			Err(err_tensor_has_stride())
		}
	}

	/// # Errors
	/// - If 'dst' doesn't have a safe map.
	pub fn nullary(o: &SliceBatch, mut f: impl FnMut(&mut T)) -> Result<()> {
		Self::ensure_safe_contiguous(o)?;
		let o = o.borrow_mut()?;
		let o = o.view_mut::<T>()?;

		unsafe {
			zip([o], [], [], |[o], [], []| f(o));
		}
		Ok(())
	}

	pub fn unary(o: &SliceBatch, a: &SliceBatch, mut f: impl FnMut(&mut T, T)) -> Result<()> {
		ensure_same_shape([o, a])?;

		Self::ensure_safe_contiguous(o)?;
		let o = o.borrow_mut()?;
		let o = o.view_mut::<T>()?;

		let a_broadcast = Self::ensure_safe_contiguous_or_broadcasted(a)?;
		let a = a.borrow()?;
		let a = a.view::<T>()?;

		unsafe {
			match a_broadcast {
				false => {
					zip([o], [a], [], |[o], [a], []| f(o, a));
				},
				_ => {
					zip([o], [], [a], |[o], [], [a]| f(o, a));
				},
			}
		}
		Ok(())
	}

	pub fn binary(
		o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, mut f: impl FnMut(&mut T, T, T),
	) -> Result<()> {
		ensure_same_shape([o, a, b])?;

		Self::ensure_safe_contiguous(o)?;
		let o = o.borrow_mut()?;
		let o = o.view_mut::<T>()?;

		let a_broadcast = Self::ensure_safe_contiguous_or_broadcasted(a)?;
		let a = a.borrow()?;
		let a = a.view::<T>()?;

		let b_broadcast = Self::ensure_safe_contiguous_or_broadcasted(b)?;
		let b = b.borrow()?;
		let b = b.view::<T>()?;

		unsafe {
			match (a_broadcast, b_broadcast) {
				(false, false) => {
					zip([o], [a, b], [], |[o], [a, b], []| f(o, a, b));
				},
				(false, _) => {
					zip([o], [a], [b], |[o], [a], [b]| f(o, a, b));
				},
				(_, false) => {
					zip([o], [b], [a], |[o], [b], [a]| f(o, a, b));
				},
				_ => {
					zip([o], [], [a, b], |[o], [], [a, b]| f(o, a, b));
				},
			}
		}
		Ok(())
	}

	pub fn n_contiguous<const N: usize>(
		t: [&SliceBatch; N], f: impl FnMut([&Cell<T>; N]),
	) -> Result<()> {
		ensure_same_shape(t)?;
		let t = array::try_map(&t, |_, t| CPUInput::new_safe_contiguous(t))?;
		unsafe {
			zip_n(t, f);
		}
		Ok(())
	}

	pub fn n_vec<const N: usize>(
		t: [&SliceBatch; N], f: impl FnMut([&[Cell<T>]; N]),
	) -> Result<()> {
		ensure_same_shape(t)?;
		let t = array::try_map(&t, |_, t| CPUInput::new_safe_contiguous(t))?;
		unsafe {
			vec_zip_n(t, f);
		}
		Ok(())
	}

	pub fn vec_reduce<const M: usize, const N: usize>(
		r: [&SliceBatch; M], a: [&SliceBatch; N], f: impl FnMut([&Cell<T>; M], [&[Cell<T>]; N]),
	) -> Result<()>
	where
		[(); 1 + N]:,
	{
		let a_shape = ensure_same_shape(a)?;
		let r_shape = ensure_same_shape(r)?;
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
		let r = array::try_map(&r, |_, r| CPUInput::new_safe_contiguous(r))?;
		let a = array::try_map(&a, |_, a| CPUInput::new_safe_contiguous(a))?;
		unsafe {
			reduce_zip_n(r, a, f);
		}
		Ok(())
	}
}

impl<T: HasDType + Copy + FromToF64> Executor for FloatExecutor<T> {
	fn read_bin(&self, dst: &SliceBatch, src: &mut dyn std::io::Read) -> Result<()> {
		let mut result = Ok(());
		Self::n_vec([dst], |[dst]| {
			if result.is_ok() {
				let ptr = dst.as_ptr() as *mut u8;
				let bytes = dst.len() * std::mem::size_of::<T>();
				let slice = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
				result = src.read_exact(slice);

				// We always store values as little-endian,
				// so conversion is needed for big-endian targets
				#[cfg(target_endian = "big")]
				{
					todo!("Reading from binary file on big-endian targets is not implemented yet");
				}
			}
		})?;
		result.map_err(Into::into)
	}

	fn write_bin(&self, src: &SliceBatch, dst: &mut dyn std::io::Write) -> Result<()> {
		#[cfg(target_endian = "big")]
		{
			todo!("Saving to binary file on big-endian targets is not implemented yet");
		}
		let mut result = Ok(());
		Self::n_vec([src], |[src]| {
			if result.is_ok() {
				let ptr = src.as_ptr() as *const u8;
				let bytes = src.len() * std::mem::size_of::<f32>();
				let slice = unsafe { std::slice::from_raw_parts(ptr, bytes) };
				result = dst.write_all(slice);
			}
		})?;
		result.map_err(Into::into)
	}

	fn zeros(&self, o: &SliceBatch) -> Result<()> {
		Self::nullary(o, |o| *o = T::from_f64(0.0))
	}

	fn randn_clamped(&self, o: &SliceBatch) -> Result<()> {
		let mut rng = self.rng.borrow_mut();
		Self::nullary(o, |o| *o = T::from_f64(rng.get_normal_clamped()))
	}

	fn copy(&self, o: &SliceBatch, a: &SliceBatch) -> Result<()> {
		Self::unary(o, a, |o, a| *o = a)
	}

	fn rsqrt(&self, o: &SliceBatch, a: &SliceBatch, scale: f64, eps: f64) -> Result<()> {
		Self::unary(o, a, |o, a| *o = T::from_f64(math::rsqrt(a.to_f64() * scale, eps)))
	}

	fn ln_clamped(&self, o: &SliceBatch, a: &SliceBatch) -> Result<()> {
		Self::unary(o, a, |o, a| *o = T::from_f64(a.to_f64().ln().max(-1000.0)))
	}

	fn add_weighted(
		&self, o: &SliceBatch, a: &SliceBatch, a_weight: f64, b: &SliceBatch, b_weight: f64,
	) -> Result<()> {
		Self::binary(o, a, b, |o, a, b| {
			*o = T::from_f64(math::add_weighted(a.to_f64(), a_weight, b.to_f64(), b_weight))
		})
	}

	fn mul(&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()> {
		Self::binary(o, a, b, |o, a, b| *o = T::from_f64(a.to_f64() * b.to_f64()))
	}

	#[allow(clippy::many_single_char_names)]
	fn mul_add(
		&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64, c: &SliceBatch,
		c_weight: f64,
	) -> Result<()> {
		/*Self::n_contiguous([o, a, b, c], |[o, a, b, c]| {
			let a = a.get().to_f64();
			let b = b.get().to_f64();
			let c = c.get().to_f64();
			let v = math::add_weighted(a * b, ab_weight, c, c_weight);
			o.set(T::from_f64(v));
		})*/
		todo!("FloatExecutor::mul_add is not implemented yet");
	}

	fn mul_acc(
		&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64, o_weight: f64,
	) -> Result<()> {
		Self::binary(o, a, b, |o, a, b| {
			*o = T::from_f64(math::add_weighted(
				a.to_f64() * b.to_f64(),
				ab_weight,
				o.to_f64(),
				o_weight,
			))
		})
	}

	fn swiglu(&self, out: &SliceBatch, lin: &SliceBatch, gate: &SliceBatch) -> Result<()> {
		Self::n_contiguous([out, lin, gate], |[out, lin, gate]| {
			let lin = lin.get().to_f64();
			let gate = gate.get().to_f64();
			let v = math::swiglu(lin, gate);
			out.set(T::from_f64(v));
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
				let (d_lin_val, d_gate_val) = math::swiglu_backward(lin, gate, d_out);
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

	fn dot(&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, scale: f64) -> Result<()> {
		Self::vec_reduce([o], [a, b], |o, [a, b]| {
			let dot = math::dot(a, b);
			let v = dot * scale;
			o[0].set(T::from_f64(v));
		})
	}

	#[allow(clippy::many_single_char_names)]
	fn dot_add(
		&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64, c: &SliceBatch,
		c_weight: f64,
	) -> Result<()> {
		Self::vec_reduce([o, c], [a, b], |[o, c], [a, b]| {
			let c = c.get().to_f64();
			let dot = math::dot(a, b);
			let v = math::add_weighted(dot, ab_weight, c, c_weight);
			o.set(T::from_f64(v));
		})
	}

	fn rsqrt_dot(
		&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, scale: f64, eps: f64,
	) -> Result<()> {
		Self::vec_reduce([o], [a, b], |o, [a, b]| {
			let dot = math::dot(a, b);
			let v = math::rsqrt(dot * scale, eps);
			o[0].set(T::from_f64(v));
		})
	}

	fn mm(&self, o: &MatrixBatch, a: &MatrixBatch, b: &MatrixBatch, scale: f64) -> Result<()> {
		//for i in 0..o.map.dims[0].size {
		//	let o = o.select(0, i)?;
		//}

		for m in o.iter_along_axis(0) {
			xyz(&m);
		}

		Ok(()) // TODO
	}
}

// TODO - delete
#[inline(never)]
fn xyz(a: &SliceBatch) {
	println!("hello world. offset = {}", a.map.offset);
}
