//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::hint::cold_path;
use std::rc::Rc;

use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut};
use crate::tensor::device::cpu::zip::{zip_elems, zip_vec_reduce, zip_vecs};
use crate::tensor::device::executor::{Executor, ensure_same_shape};
use crate::tensor::generic::buffer::Buffer;
use crate::tensor::generic::map::{ND, NDShape};
use crate::tensor::{HasDType, generic};
use crate::{Error, Result};

use super::math::{self, FromToF64};
use super::rng::Rng;

//--------------------------------------------------------------------------------------------------

#[cold]
#[inline(never)]
fn err_tensor_has_stride() -> Error {
	"Tensor data is neither contiguous nor broadcasted.".into()
}

#[cold]
#[inline(never)]
fn err_tensor_not_contiguous() -> Error {
	"Tensor data is not contiguous.".into()
}

#[cold]
#[inline(never)]
fn err_tensor_invalid_shape(shape: [usize; 2], expected: [usize; 2]) -> Error {
	format!("Tensor shape {:?} does not match expected shape {:?}", shape, expected).into()
}

fn ensure_expected_shape<B: Buffer>(
	tensor: &generic::Tensor<ND<2>, B>,
	expected: [usize; 2],
) -> Result<()> {
	let shape = tensor.map.nd_shape()?;
	if shape != expected {
		cold_path();
		return Err(err_tensor_invalid_shape(shape, expected));
	}
	Ok(())
}

pub struct FloatExecutor<T: Copy + HasDType + FromToF64> {
	rng: Rc<RefCell<Rng>>,
	phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + HasDType + FromToF64> FloatExecutor<T> {
	pub fn new(rng: Rc<RefCell<Rng>>) -> Self {
		Self { rng, phantom: std::marker::PhantomData }
	}

	pub fn view_contiguous<'a>(
		tensor: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
	) -> Result<generic::Tensor<ND<2>, &'a [T]>>
	where
		T: 'a,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map.dims[1];
		if !feature_dim.is_contiguous() {
			cold_path();
			return Err(err_tensor_not_contiguous());
		}
		tensor.view()
	}

	pub fn view_contiguous_mut<'a, 'b>(
		tensor: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'b>>,
	) -> Result<generic::Tensor<ND<2>, &'b mut [T]>>
	where
		T: 'b,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map.dims[1];
		if !feature_dim.is_contiguous() {
			cold_path();
			return Err(err_tensor_not_contiguous());
		}
		tensor.view_mut()
	}

	pub fn view_contiguous_or_broadcasted<'a>(
		tensor: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
	) -> Result<(generic::Tensor<ND<2>, &'a [T]>, bool)>
	where
		T: 'a,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map.dims[1];
		let broadcast = if feature_dim.is_contiguous() {
			false
		} else if feature_dim.is_broadcasted() {
			true
		} else {
			cold_path();
			return Err(err_tensor_has_stride());
		};
		Ok((tensor.view()?, broadcast))
	}

	pub fn view_contiguous_or_broadcasted_mut<'a>(
		tensor: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
	) -> Result<(generic::Tensor<ND<2>, &'a mut [T]>, bool)>
	where
		T: 'a,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map.dims[1];
		let broadcast = if feature_dim.is_contiguous() {
			false
		} else if feature_dim.is_broadcasted() {
			true
		} else {
			cold_path();
			return Err(err_tensor_has_stride());
		};
		Ok((tensor.view_mut()?, broadcast))
	}

	pub fn nullary<'a, 'b>(
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'b>>,
		mut f: impl FnMut(&mut T),
	) -> Result<()>
	where
		T: 'b,
	{
		let o = Self::view_contiguous_mut(o)?;
		unsafe {
			zip_elems([o], [], [], |[o], [], []| f(o));
		}
		Ok(())
	}

	pub fn unary<'a>(
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		mut f: impl FnMut(&mut T, T),
	) -> Result<()>
	where
		T: 'a,
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

	pub fn binary<'a>(
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		b: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		mut f: impl FnMut(&mut T, T, T),
	) -> Result<()>
	where
		T: 'a,
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

impl<T: HasDType + Copy + FromToF64> Executor for FloatExecutor<T> {
	fn read_bin<'a>(
		&'a self,
		dst: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		src: &'a mut dyn std::io::Read,
	) -> Result<()> {
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

	fn write_bin(
		&self,
		src: &generic::Tensor<ND<2>, DeviceBufferRef>,
		dst: &mut dyn std::io::Write,
	) -> Result<()> {
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

	fn zeros<'a, 'b, 'c>(
		&'a self,
		o: &'b mut generic::Tensor<ND<2>, DeviceBufferRefMut<'c>>,
	) -> Result<()> {
		Self::nullary(o, |o| *o = T::from_f64(0.0))
	}

	fn randn_clamped<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
	) -> Result<()> {
		let mut rng = self.rng.borrow_mut();
		Self::nullary(o, |o| *o = T::from_f64(rng.get_normal_clamped()))
	}

	fn copy<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
	) -> Result<()> {
		Self::unary(o, a, |o, a| *o = a)
	}

	fn rsqrt<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		scale: f64,
		eps: f64,
	) -> Result<()> {
		Self::unary(o, a, |o, a| *o = T::from_f64(math::rsqrt(a.to_f64() * scale, eps)))
	}

	fn ln_clamped<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
	) -> Result<()> {
		Self::unary(o, a, |o, a| *o = T::from_f64(a.to_f64().ln().max(-1000.0)))
	}

	fn add_weighted(
		&self,
		o: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef>,
		a_weight: f64,
		b: &generic::Tensor<ND<2>, DeviceBufferRef>,
		b_weight: f64,
	) -> Result<()> {
		Self::binary(o, a, b, |o, a, b| {
			*o = T::from_f64(math::add_weighted(a.to_f64(), a_weight, b.to_f64(), b_weight))
		})
	}

	fn mul(
		&self,
		o: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef>,
	) -> Result<()> {
		Self::binary(o, a, b, |o, a, b| *o = T::from_f64(a.to_f64() * b.to_f64()))
	}

	#[allow(clippy::many_single_char_names)]
	fn mul_add(
		&self,
		_o: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		_a: &generic::Tensor<ND<2>, DeviceBufferRef>,
		_b: &generic::Tensor<ND<2>, DeviceBufferRef>,
		_ab_weight: f64,
		_c: &generic::Tensor<ND<2>, DeviceBufferRef>,
		_c_weight: f64,
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
		&self,
		o: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef>,
		ab_weight: f64,
		o_weight: f64,
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

	fn swiglu(
		&self,
		out: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		lin: &generic::Tensor<ND<2>, DeviceBufferRef>,
		gate: &generic::Tensor<ND<2>, DeviceBufferRef>,
	) -> Result<()> {
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

	fn swiglu_backward(
		&self,
		_d_lin: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		_d_gate: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		_lin: &generic::Tensor<ND<2>, DeviceBufferRef>,
		_gate: &generic::Tensor<ND<2>, DeviceBufferRef>,
		_d_out: &generic::Tensor<ND<2>, DeviceBufferRef>,
	) -> Result<()> {
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

	fn sum_all(&self, a: &generic::Tensor<ND<2>, DeviceBufferRef>) -> Result<f64> {
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

	fn approx_eq(
		&self,
		a: &generic::Tensor<ND<2>, DeviceBufferRef>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef>,
		eps: f64,
	) -> Result<bool> {
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

	fn softmax(
		&self,
		out: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		inp: &generic::Tensor<ND<2>, DeviceBufferRef>,
	) -> Result<()> {
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

	fn dot(
		&self,
		o: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef>,
		scale: f64,
	) -> Result<()> {
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
	fn dot_add(
		&self,
		_o: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		_a: &generic::Tensor<ND<2>, DeviceBufferRef>,
		_b: &generic::Tensor<ND<2>, DeviceBufferRef>,
		_ab_weight: f64,
		_c: &generic::Tensor<ND<2>, DeviceBufferRef>,
		_c_weight: f64,
	) -> Result<()> {
		/*Self::vec_reduce([o, c], [a, b], |[o, c], [a, b]| {
			let c = c.get().to_f64();
			let dot = math::dot(a, b);
			let v = math::add_weighted(dot, ab_weight, c, c_weight);
			o.set(T::from_f64(v));
		})*/
		todo!("FloatExecutor::dot_add is not implemented yet");
	}

	fn rsqrt_dot(
		&self,
		o: &generic::Tensor<ND<2>, DeviceBufferRefMut>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef>,
		scale: f64,
		eps: f64,
	) -> Result<()> {
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

	fn mm(
		&self,
		_o: &generic::Tensor<ND<3>, DeviceBufferRefMut>,
		_a: &generic::Tensor<ND<3>, DeviceBufferRef>,
		_b: &generic::Tensor<ND<3>, DeviceBufferRef>,
		_scale: f64,
	) -> Result<()> {
		//for i in 0..o.map.dims[0].size {
		//	let o = o.select(0, i)?;
		//}

		//for m in o.iter_along_axis(0) {
		//	xyz(&m);
		//}

		Ok(()) // TODO
	}
}
