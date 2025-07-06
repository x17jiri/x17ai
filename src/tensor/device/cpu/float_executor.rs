//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::hint::cold_path;
use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut};
use crate::tensor::device::cpu::zip::{zip_elems, zip_vec_reduce, zip_vecs, zip_vecs_varsize};
use crate::tensor::device::executor::{Executor, ExecutorError, ensure_same_shape};
use crate::tensor::generic::buffer::Buffer;
use crate::tensor::generic::map::{Map, ND, Select};
use crate::tensor::{HasDType, generic};

use super::math::{self, FromToF64};
use super::rng::Rng;

//--------------------------------------------------------------------------------------------------

trait Slice2D<T> {
	fn slice(&self, dim0: usize, dim1: std::ops::RangeFull) -> &[T];
}

impl<T: Copy + HasDType + FromToF64> Slice2D<T> for generic::Tensor<ND<2>, &[T]> {
	fn slice(&self, dim0: usize, _dim1: std::ops::RangeFull) -> &[T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		self.buf().get(span).unwrap()
	}
}

trait Slice3D<T> {
	fn slice(&self, dim0: usize, dim1: usize, dim2: std::ops::RangeFull) -> &[T];
}

impl<T: Copy + HasDType + FromToF64> Slice3D<T> for generic::Tensor<ND<3>, &[T]> {
	fn slice(&self, dim0: usize, dim1: usize, _dim2: std::ops::RangeFull) -> &[T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let map = map.select(0, dim1).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		self.buf().get(span).unwrap()
	}
}

trait SliceMut2D<T> {
	fn slice_mut(&mut self, dim0: usize, dim1: std::ops::RangeFull) -> &mut [T];
}

impl<T: Copy + HasDType + FromToF64> SliceMut2D<T> for generic::Tensor<ND<2>, &mut [T]> {
	fn slice_mut(&mut self, dim0: usize, dim1: std::ops::RangeFull) -> &mut [T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		unsafe { self.buf_mut() }.get_mut(span).unwrap()
	}
}

trait SliceMut3D<T> {
	fn slice_mut(&mut self, dim0: usize, dim1: usize, dim2: std::ops::RangeFull) -> &mut [T];
}

impl<T: Copy + HasDType + FromToF64> SliceMut3D<T> for generic::Tensor<ND<3>, &mut [T]> {
	fn slice_mut(&mut self, dim0: usize, dim1: usize, dim2: std::ops::RangeFull) -> &mut [T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let map = map.select(0, dim1).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		unsafe { self.buf_mut() }.get_mut(span).unwrap()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct ContiguousOutput<'t, T> {
	pub tensor: generic::Tensor<&'t ND<2>, &'t mut [T]>,
}

pub struct ContiguousInput<'t, T> {
	pub tensor: generic::Tensor<&'t ND<2>, &'t [T]>,
}

pub struct BroadcastedInput<'t, T> {
	pub tensor: generic::Tensor<&'t ND<2>, &'t [T]>,
}

pub enum Input<'t, T> {
	Contiguous(ContiguousInput<'t, T>),
	Broadcasted(BroadcastedInput<'t, T>),
}

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

impl<T: 'static + Copy + HasDType + FromToF64> FloatExecutor<T> {
	pub fn new(rng: Rc<RefCell<Rng>>) -> Self {
		Self { rng, phantom: std::marker::PhantomData }
	}

	pub fn view_contiguous<'t, 'buf>(
		tensor: &'t generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<ContiguousInput<'t, T>, ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		if !feature_dim.is_contiguous() {
			return Err(ExecutorError::not_contiguous());
		}
		Ok(ContiguousInput { tensor: tensor.view()? })
	}

	pub fn view_contiguous_mut<'t, 'buf>(
		tensor: &'t mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<ContiguousOutput<'t, T>, ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		if !feature_dim.is_contiguous() {
			return Err(ExecutorError::not_contiguous());
		}
		Ok(ContiguousOutput { tensor: tensor.view_mut()? })
	}

	pub fn view_contiguous_or_broadcasted<'t, 'buf>(
		tensor: &'t generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<Input<'t, T>, ErrPack<ExecutorError>>
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
		let view = tensor.view()?;
		if broadcast {
			Ok(Input::Broadcasted(BroadcastedInput { tensor: view }))
		} else {
			Ok(Input::Contiguous(ContiguousInput { tensor: view }))
		}
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
		let a = Self::view_contiguous_or_broadcasted(a)?;
		unsafe {
			match a {
				Input::Contiguous(a) => {
					zip_elems([o], [a], [], |[o], [a], []| f(o, a));
				},
				Input::Broadcasted(a) => {
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
		let a = Self::view_contiguous_or_broadcasted(a)?;
		let b = Self::view_contiguous_or_broadcasted(b)?;
		unsafe {
			match (a, b) {
				(Input::Contiguous(a), Input::Contiguous(b)) => {
					zip_elems([o], [a, b], [], |[o], [a, b], []| f(o, a, b));
				},
				(Input::Contiguous(a), Input::Broadcasted(b)) => {
					zip_elems([o], [a], [b], |[o], [a], [b]| f(o, a, b));
				},
				(Input::Broadcasted(a), Input::Contiguous(b)) => {
					zip_elems([o], [b], [a], |[o], [b], [a]| f(o, a, b));
				},
				(Input::Broadcasted(a), Input::Broadcasted(b)) => {
					zip_elems([o], [], [a, b], |[o], [], [a, b]| f(o, a, b));
				},
			}
		}
		Ok(())
	}

	pub fn attention_tile<const FIRST: bool>(
		acc: &mut generic::Tensor<ND<3>, &mut [f64]>, // [output, head, vo_feature]

		q: &generic::Tensor<ND<3>, &[T]>, // [output, head, qk_feature]
		k: &generic::Tensor<ND<3>, &[T]>, // [input, head, qk_feature]
		v: &generic::Tensor<ND<3>, &[T]>, // [input, head, vo_feature]

		// `acc`, `prev_max` and `prev_sum` will be initialized when processing the first tile.
		prev_max: &mut generic::Tensor<ND<2>, &mut [f64]>, // [output, head]
		prev_sum: &mut generic::Tensor<ND<2>, &mut [f64]>, // [output, head]

		// Scratch space for storing scores. It doesn't need to be initialized.
		// On GPU, its shape will be [output, head, input]. However, we process outputs
		// sequentially, so we don't need separate space for each output.
		scores: &mut generic::Tensor<ND<2>, &mut [f64]>, // [head, input]
	) {
		let O = q.size(0).unwrap();
		let I = k.size(0).unwrap();
		let H = q.size(1).unwrap();
		let VO = v.size(2).unwrap();
		for j in 0..O {
			for i in 0..I {
				for h in 0..H {
					let q = q.slice(j, h, ..);
					let k = k.slice(i, h, ..);
					scores[[h, i]] = math::dot(q, k);
					// scores[h][i].set(math::dot(q, k))
				}
			}
			if FIRST {
				for h in 0..H {
					let prev_max = &mut prev_max[[j, h]];
					let prev_sum = &mut prev_sum[[j, h]];

					let scores = scores.slice_mut(h, ..);
					let scores = &mut scores[..I]; // TODO
					let (first_max, first_sum) = math::softmax_part1_(scores);

					*prev_max = first_max;
					*prev_sum = first_sum;

					let acc = acc.slice_mut(j, h, ..);
					for i in 0..1 {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64();
						for f in 0..VO {
							acc[f] = score * v[f].to_f64();
						}
					}
					for i in 1..I {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64();
						for f in 0..VO {
							acc[f] += score * v[f].to_f64();
						}
					}
				}
			} else {
				for h in 0..H {
					let prev_max = &mut prev_max[[j, h]];
					let prev_sum = &mut prev_sum[[j, h]];

					let scores = scores.slice_mut(h, ..);
					let scores = &mut scores[..I]; // TODO
					let (new_max, new_sum) = math::softmax_part1_(scores);

					let total_max = prev_max.max(new_max);
					*prev_max = total_max;

					let prev_weight = (*prev_max - total_max).exp();
					let new_weight = (new_max - total_max).exp();

					*prev_sum = (*prev_sum * prev_weight) + (new_sum * new_weight);

					let acc = acc.slice_mut(j, h, ..);
					for i in 0..1 {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64() * new_weight;
						for f in 0..VO {
							acc[f] = (acc[f] * prev_weight) + (score * v[f].to_f64());
						}
					}
					for i in 1..I {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64() * new_weight;
						for f in 0..VO {
							acc[f] += score * v[f].to_f64();
						}
					}
				}
			}
		}
	}
}

impl<T: 'static + HasDType + Copy + FromToF64> Executor for FloatExecutor<T> {
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
					let bytes = std::mem::size_of_val(dst);
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
					let bytes = std::mem::size_of_val(src);
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
			));
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

	#[allow(clippy::panic_in_result_fn)]
	fn swiglu_backward<'buf>(
		&self,
		d_lin_gate: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		swapped: bool,
		lin: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		gate: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		d_out: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		let shape = ensure_same_shape([], [lin, gate, d_out])?;
		let d_lin_gate_shape = d_lin_gate.nd_shape()?;
		if d_lin_gate_shape[0] != shape[0] || d_lin_gate_shape[1] / 2 < shape[1] {
			cold_path();
			return Err(ErrPack {
				code: ExecutorError::InvalidShape,
				extra: None,
			});
		}

		let d_lin_gate = Self::view_contiguous_mut(d_lin_gate)?;
		let lin = Self::view_contiguous(lin)?;
		let gate = Self::view_contiguous(gate)?;
		let d_out = Self::view_contiguous(d_out)?;

		let (d_lin_start, d_gate_start) = if swapped {
			(d_lin_gate_shape[1] - shape[1], 0)
		} else {
			(0, d_lin_gate_shape[1] - shape[1])
		};
		unsafe {
			zip_vecs_varsize(
				[d_lin_gate],
				[lin, gate, d_out],
				|[d_lin_gate], [lin, gate, d_out]| {
					for i in 0..shape[1] {
						let lin = lin[i].to_f64();
						let gate = gate[i].to_f64();
						let d_out = d_out[i].to_f64();
						let (d_lin_val, d_gate_val) = math::swiglu_backward(lin, gate, d_out);
						d_lin_gate[d_lin_start + i] = T::from_f64(d_lin_val);
						d_lin_gate[d_gate_start + i] = T::from_f64(d_gate_val);
					}
				},
			);
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

	fn dot_acc<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		o_weight: f64,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		let shape = ensure_same_shape([], [a, b])?;
		ensure_expected_shape(o, [shape[0], 1])?;
		let o = Self::view_contiguous_mut(o)?;
		let a = Self::view_contiguous(a)?;
		let b = Self::view_contiguous(b)?;
		unsafe {
			zip_vec_reduce(o, [a, b], |o, [a, b]| {
				*o = T::from_f64(math::add_weighted(
					o.to_f64(),
					o_weight,
					math::dot(a, b),
					ab_weight,
				));
			});
		}
		Ok(())
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

	fn sqrt_dot<'buf>(
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
			zip_vec_reduce(o, [a, b], |o, [a, b]| {
				let dot = math::dot(a, b);
				*o = T::from_f64((dot * scale).sqrt());
			});
		}
		Ok(())
	}

	#[allow(clippy::panic_in_result_fn)]
	#[allow(clippy::many_single_char_names)]
	fn mm<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		let m = o.map().dims[0].size;
		let n = o.map().dims[1].size;
		let k = a.map().dims[1].size;

		assert!(a.map().dims[0].size == m);
		assert!(b.map().dims[0].size == k);
		assert!(b.map().dims[1].size == n);

		o.ensure_safe()?;
		let o_row_stride = o.map().dims[0].stride;
		let o_col_stride = o.map().dims[1].stride;
		let mut o = o.view_mut::<T>()?;
		let o_off = o.map().offset;
		let o = unsafe { &mut o.buf_mut()[o_off..] };

		a.ensure_safe()?;
		let a_row_stride = a.map().dims[0].stride;
		let a_col_stride = a.map().dims[1].stride;
		let a = a.view::<T>()?;
		let a = &a.buf()[a.map().offset..];

		b.ensure_safe()?;
		let b_row_stride = b.map().dims[0].stride;
		let b_col_stride = b.map().dims[1].stride;
		let b = b.view::<T>()?;
		let b = &b.buf()[b.map().offset..];

		for j in 0..m {
			for i in 0..n {
				let mut t = 0.0;
				for k in 0..k {
					let a = a[j * a_row_stride + k * a_col_stride];
					let b = b[k * b_row_stride + i * b_col_stride];
					t += a.to_f64() * b.to_f64();
				}
				let t = T::from_f64(t * scale);
				o[j * o_row_stride + i * o_col_stride] = t;
			}
		}

		Ok(()) // TODO
	}

	fn attention(
		&self,
		o: &mut generic::Tensor<ND<3>, DeviceBufferRefMut>, // [inputs, qo_heads, vo_features]
		q: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, qo_heads, qk_features]
		k: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, k_heads, qk_features]
		v: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, v_heads, vo_features]
	) {
		todo!("FloatExecutor::attention is not implemented yet");
	}
}
