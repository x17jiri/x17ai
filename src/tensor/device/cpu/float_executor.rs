//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut};
use crate::tensor::device::cpu::zip::{zip_elems, zip_vec_reduce, zip_vecs};
use crate::tensor::device::executor::{Executor, ExecutorError, ensure_same_shape};
use crate::tensor::generic::buffer::Buffer;
use crate::tensor::generic::map::ND;
use crate::tensor::{HasDType, generic};

use super::math::{self, FromToF64};
use super::rng::Rng;

//--------------------------------------------------------------------------------------------------

fn ensure_expected_shape<B: Buffer>(
	tensor: &generic::Tensor<ND<2>, B>,
	expected: [usize; 2],
) -> Result<(), ErrPack<ExecutorError>> {
	let shape = tensor.nd_shape()?;
	if shape != expected {
		return Err(ExecutorError::invalid_shape(shape, expected));
	}
	Ok(())
}

pub struct FloatExecutor<T: Copy + HasDType + FromToF64> {
	rng: Rc<RefCell<Rng>>,
	phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + HasDType + FromToF64> FloatExecutor<T>
where
	T: 'static,
{
	pub fn new(rng: Rc<RefCell<Rng>>) -> Self {
		Self { rng, phantom: std::marker::PhantomData }
	}

	pub fn view_contiguous<'t, 'buf>(
		tensor: &'t generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<generic::Tensor<ND<2>, &'t [T]>, ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		if !feature_dim.is_contiguous() {
			return Err(ExecutorError::not_contiguous());
		}
		Ok(tensor.view()?)
	}

	pub fn view_contiguous_mut<'t, 'buf>(
		tensor: &'t mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<generic::Tensor<ND<2>, &'t mut [T]>, ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		if !feature_dim.is_contiguous() {
			return Err(ExecutorError::not_contiguous());
		}
		Ok(tensor.view_mut()?)
	}

	pub fn view_contiguous_or_broadcasted<'t, 'buf>(
		tensor: &'t generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(generic::Tensor<ND<2>, &'t [T]>, bool), ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		let broadcast = if feature_dim.is_contiguous() {
			false
		} else if feature_dim.is_broadcasted() {
			true
		} else {
			return Err(ExecutorError::not_contiguous_or_broadcasted());
		};
		Ok((tensor.view()?, broadcast))
	}

	pub fn view_contiguous_or_broadcasted_mut<'t, 'buf>(
		tensor: &'t mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(generic::Tensor<ND<2>, &'t mut [T]>, bool), ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		let broadcast = if feature_dim.is_contiguous() {
			false
		} else if feature_dim.is_broadcasted() {
			true
		} else {
			return Err(ExecutorError::not_contiguous_or_broadcasted());
		};
		Ok((tensor.view_mut()?, broadcast))
	}

	pub fn nullary<'buf>(
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		mut f: impl FnMut(&mut T),
	) -> Result<(), ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		let o = Self::view_contiguous_mut(o)?;
		unsafe {
			zip_elems([o], [], [], |[o], [], []| f(o));
		}
		Ok(())
	}

	pub fn unary<'buf>(
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		mut f: impl FnMut(&mut T, T),
	) -> Result<(), ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		ensure_same_shape([o], [a])?;
		let o = Self::view_contiguous_mut(o)?;
		let (a, a_broadcast) = Self::view_contiguous_or_broadcasted(a)?;
		unsafe {
			match a_broadcast {
				false => {
					zip_elems([o], [a], [], |[o], [a], []| f(o, a));
				},
				_ => {
					zip_elems([o], [], [a], |[o], [], [a]| f(o, a));
				},
			}
		}
		Ok(())
	}

	pub fn binary<'buf>(
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		mut f: impl FnMut(&mut T, T, T),
	) -> Result<(), ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		ensure_same_shape([o], [a, b])?;
		let o = Self::view_contiguous_mut(o)?;
		let (a, a_broadcast) = Self::view_contiguous_or_broadcasted(a)?;
		let (b, b_broadcast) = Self::view_contiguous_or_broadcasted(b)?;
		unsafe {
			match (a_broadcast, b_broadcast) {
				(false, false) => {
					zip_elems([o], [a, b], [], |[o], [a, b], []| f(o, a, b));
				},
				(false, _) => {
					zip_elems([o], [a], [b], |[o], [a], [b]| f(o, a, b));
				},
				(_, false) => {
					zip_elems([o], [b], [a], |[o], [b], [a]| f(o, a, b));
				},
				_ => {
					zip_elems([o], [], [a, b], |[o], [], [a, b]| f(o, a, b));
				},
			}
		}
		Ok(())
	}
}

impl<T: HasDType + Copy + FromToF64> Executor for FloatExecutor<T>
where
	T: 'static,
{
	fn read_bin<'buf>(
		&self,
		dst: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		src: &mut dyn std::io::Read,
	) -> Result<(), ErrPack<ExecutorError>> {
		let mut result = Ok(());
		let dst = Self::view_contiguous_mut(dst)?;
		unsafe {
			zip_vecs([dst], [], |[dst], []| {
				if result.is_ok() {
					let ptr = dst.as_mut_ptr().cast();
					let bytes = dst.len() * std::mem::size_of::<T>();
					let slice = std::slice::from_raw_parts_mut(ptr, bytes);
					result = src.read_exact(slice);

					// We always store values as little-endian,
					// so conversion is needed for big-endian targets
					#[cfg(target_endian = "big")]
					{
						todo!(
							"Reading from binary file on big-endian targets is not implemented yet"
						);
					}
				}
			});
		}
		result.map_err(Into::into)
	}

	fn write_bin<'buf>(
		&self,
		src: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		dst: &mut dyn std::io::Write,
	) -> Result<(), ErrPack<ExecutorError>> {
		#[cfg(target_endian = "big")]
		{
			todo!("Saving to binary file on big-endian targets is not implemented yet");
		}
		let mut result = Ok(());
		let src = Self::view_contiguous(src)?;
		unsafe {
			zip_vecs([], [src], |[], [src]| {
				if result.is_ok() {
					let ptr = src.as_ptr().cast();
					let bytes = src.len() * std::mem::size_of::<T>();
					let slice = std::slice::from_raw_parts(ptr, bytes);
					result = dst.write_all(slice);
				}
			});
		}
		result.map_err(Into::into)
	}

	fn zeros<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		Self::nullary(o, |o| *o = T::from_f64(0.0))
	}

	fn randn_clamped<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		let mut rng = self.rng.borrow_mut();
		Self::nullary(o, |o| *o = T::from_f64(rng.get_normal_clamped()))
	}

	fn copy<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		Self::unary(o, a, |o, a| *o = a)
	}

	fn rsqrt<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
		eps: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		Self::unary(o, a, |o, a| *o = T::from_f64(math::rsqrt(a.to_f64() * scale, eps)))
	}

	fn ln_clamped<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		Self::unary(o, a, |o, a| *o = T::from_f64(a.to_f64().ln().max(-1000.0)))
	}

	fn add_weighted<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		a_weight: f64,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		Self::binary(o, a, b, |o, a, b| {
			*o = T::from_f64(math::add_weighted(a.to_f64(), a_weight, b.to_f64(), b_weight));
		})
	}

	fn acc_weighted<'buf>(
		&self,
		a: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a_weight: f64,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		Self::unary(a, b, |a, b| {
			*a = T::from_f64(math::add_weighted(a.to_f64(), a_weight, b.to_f64(), b_weight));
		})
	}

	fn mul<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		Self::binary(o, a, b, |o, a, b| *o = T::from_f64(a.to_f64() * b.to_f64()))
	}

	fn mul_<'buf>(
		&self,
		a: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		Self::unary(a, b, |a, b| *a = T::from_f64(a.to_f64() * b.to_f64()))
	}

	#[allow(clippy::many_single_char_names)]
	fn mul_add<'buf>(
		&self,
		_o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		_a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_ab_weight: f64,
		_c: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_c_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		/*Self::n_contiguous([o, a, b, c], |[o, a, b, c]| {
			let a = a.get().to_f64();
			let b = b.get().to_f64();
			let c = c.get().to_f64();
			let v = math::add_weighted(a * b, ab_weight, c, c_weight);
			o.set(T::from_f64(v));
		})*/
		todo!("FloatExecutor::mul_add is not implemented yet");
	}

	fn mul_acc<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
		o_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		Self::binary(o, a, b, |o, a, b| {
			*o = T::from_f64(math::add_weighted(
				a.to_f64() * b.to_f64(),
				ab_weight,
				o.to_f64(),
				o_weight,
			))
		})
	}

	fn swiglu<'buf>(
		&self,
		out: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		lin: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		gate: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		ensure_same_shape([out], [lin, gate])?;
		let out = Self::view_contiguous_mut(out)?;
		let lin = Self::view_contiguous(lin)?;
		let gate = Self::view_contiguous(gate)?;
		unsafe {
			zip_elems([out], [lin, gate], [], |[out], [lin, gate], []| {
				*out = T::from_f64(math::swiglu(lin.to_f64(), gate.to_f64()));
			});
		}
		Ok(())
	}

	fn swiglu_backward<'buf>(
		&self,
		_d_lin: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		_d_gate: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		_lin: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_gate: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_d_out: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		/*Self::n_contiguous(
			[d_lin, d_gate, lin, gate, d_out],
			|[d_lin, d_gate, lin, gate, d_out]| {
				let lin = lin.get().to_f64();
				let gate = gate.get().to_f64();
				let d_out = d_out.get().to_f64();
				let (d_lin_val, d_gate_val) = math::swiglu_backward(lin, gate, d_out);
				d_lin.set(T::from_f64(d_lin_val));
				d_gate.set(T::from_f64(d_gate_val));
			},
		)*/
		todo!("FloatExecutor::swiglu_backward is not implemented yet");
	}

	#[allow(clippy::panic_in_result_fn)]
	fn swiglu_backward2<'buf>(
		&self,
		d_lin_gate: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		size: usize,
		swapped: bool,
		lin: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		gate: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		d_out: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		// TODO - ensure shapes

		let d_lin_gate = Self::view_contiguous_mut(d_lin_gate)?;
		let lin = Self::view_contiguous(lin)?;
		let gate = Self::view_contiguous(gate)?;
		let d_out = Self::view_contiguous(d_out)?;

		let dim_size = d_lin_gate.map().dims[0].size;
		assert!(size <= dim_size / 2);
		let (d_lin_start, d_gate_start) =
			if swapped { (dim_size - size, 0) } else { (0, dim_size - size) };
		unsafe {
			zip_vecs([d_lin_gate], [lin, gate, d_out], |[d_lin_gate], [lin, gate, d_out]| {
				for i in 0..size {
					let lin = lin[i].to_f64();
					let gate = gate[i].to_f64();
					let d_out = d_out[i].to_f64();
					let (d_lin_val, d_gate_val) = math::swiglu_backward(lin, gate, d_out);
					d_lin_gate[d_lin_start + i] = T::from_f64(d_lin_val);
					d_lin_gate[d_gate_start + i] = T::from_f64(d_gate_val);
				}
			});
		}
		Ok(())
	}

	fn sum_all<'buf>(
		&self,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<f64, ErrPack<ExecutorError>> {
		// TODO - this could handle broadcasted tensors as well
		let a = Self::view_contiguous(a)?;
		let mut sum = 0.0;
		unsafe {
			zip_elems([], [a], [], |[], [a], []| {
				sum += a.to_f64();
			});
		}
		Ok(sum)
	}

	fn approx_eq<'buf>(
		&self,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		eps: f64,
	) -> Result<bool, ErrPack<ExecutorError>> {
		// TODO - this could handle broadcasted tensors as well
		ensure_same_shape([], [a, b])?;
		let a = Self::view_contiguous(a)?;
		let b = Self::view_contiguous(b)?;
		let mut result = true;
		unsafe {
			zip_elems([], [a, b], [], |[], [a, b], []| {
				result &= math::approx_eq(a.to_f64(), b.to_f64(), eps);
			});
		}
		Ok(result)
	}

	fn softmax<'buf>(
		&self,
		out: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		inp: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		ensure_same_shape([out], [inp])?;
		let out = Self::view_contiguous_mut(out)?;
		let inp = Self::view_contiguous(inp)?;
		unsafe {
			zip_vecs([out], [inp], |[out], [inp]| {
				let (_max, sum) = math::softmax_part1(inp, out);
				math::softmax_part2_(sum, out);
			});
		}
		Ok(())
	}

	fn softmax_<'buf>(
		&self,
		t: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		let t = Self::view_contiguous_mut(t)?;
		unsafe {
			zip_vecs([t], [], |[t], []| {
				let (_max, sum) = math::softmax_part1_(t);
				math::softmax_part2_(sum, t);
			});
		}
		Ok(())
	}

	fn dot<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		let shape = ensure_same_shape([], [a, b])?;
		ensure_expected_shape(o, [shape[0], 1])?;
		let o = Self::view_contiguous_mut(o)?;
		let a = Self::view_contiguous(a)?;
		let b = Self::view_contiguous(b)?;
		unsafe {
			zip_vec_reduce(o, [a, b], |o, [a, b]| *o = T::from_f64(math::dot(a, b) * scale));
		}
		Ok(())
	}

	#[allow(clippy::many_single_char_names)]
	fn dot_add<'buf>(
		&self,
		_o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		_a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_ab_weight: f64,
		_c: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_c_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		/*Self::vec_reduce([o, c], [a, b], |[o, c], [a, b]| {
			let c = c.get().to_f64();
			let dot = math::dot(a, b);
			let v = math::add_weighted(dot, ab_weight, c, c_weight);
			o.set(T::from_f64(v));
		})*/
		todo!("FloatExecutor::dot_add is not implemented yet");
	}

	fn rsqrt_dot<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
		eps: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		let shape = ensure_same_shape([], [a, b])?;
		ensure_expected_shape(o, [shape[0], 1])?;
		let o = Self::view_contiguous_mut(o)?;
		let a = Self::view_contiguous(a)?;
		let b = Self::view_contiguous(b)?;
		unsafe {
			zip_vec_reduce(o, [a, b], |o, [a, b]| {
				let dot = math::dot(a, b);
				*o = T::from_f64(math::rsqrt(dot * scale, eps));
			});
		}
		Ok(())
	}

	fn mm<'buf>(
		&self,
		_o: &mut generic::Tensor<ND<3>, DeviceBufferRefMut<'buf>>,
		_a: &generic::Tensor<ND<3>, DeviceBufferRef<'buf>>,
		_b: &generic::Tensor<ND<3>, DeviceBufferRef<'buf>>,
		_scale: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		//for i in 0..o.map.dims[0].size {
		//	let o = o.select(0, i)?;
		//}

		//for m in o.iter_along_axis(0) {
		//	xyz(&m);
		//}

		Ok(()) // TODO
	}
}
