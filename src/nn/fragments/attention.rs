//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::hint::{cold_path, likely};
use std::rc::Rc;

use smallvec::{SmallVec, smallvec};

use crate::autograd::{AutogradTensor, BackwardFn};
use crate::nn::Param;
use crate::nn::fragments::{Fragment, UnaryFragment};
use crate::rng::Rng;
use crate::tensor::device::buffer::AttentionArgs;
use crate::tensor::generic::map::dd::{DimVecBuilder, INLINE_DIMS};
use crate::tensor::generic::map::{Map, SizeAndStride};
use crate::tensor::{Tensor, TensorOpError};
use crate::util::mycell::UnsafeBorrowFailFlag;
use crate::{ErrPack, autograd};

//--------------------------------------------------------------------------------------------------

pub struct Attention;

impl Default for Attention {
	fn default() -> Self {
		Self::new()
	}
}

impl Attention {
	pub fn new() -> Self {
		Self {}
	}
}

impl Attention {
	#[inline(never)]
	fn alloc_output(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, ErrPack<TensorOpError>> {
		let q_ndim = q.ndim();
		let k_ndim = k.ndim();
		let v_ndim = v.ndim();
		if q_ndim < 3 || k_ndim < 3 || v_ndim < 3 {
			cold_path();
			return Err(ErrPack {
				code: TensorOpError::NotEnoughDimensions,
				extra: None,
			});
		}
		let ndim = q_ndim.max(k_ndim).max(v_ndim);

		// TODO - could rewrite using DimVecBuilder
		let mut shape: SmallVec<[usize; INLINE_DIMS]> = smallvec![0; ndim];
		let shape = shape.as_mut_slice();
		shape[ndim - 2] = q.size(-2).unwrap();
		shape[ndim - 1] = v.size(-1).unwrap();
		let mut can_reuse = shape[ndim - 1] == q.size(-1).unwrap();
		for i in 0..ndim - 2 {
			let q_size = q.size(i).unwrap_or(1);
			let k_size = k.size(i).unwrap_or(1);
			let v_size = v.size(i).unwrap_or(1);
			shape[i] = q_size.max(k_size).max(v_size);
			can_reuse &= shape[i] == q_size;
		}

		if can_reuse {
			q.reuse_or_new_like()
		} else {
			cold_path();
			q.new_empty(shape, q.dtype())
		}
	}

	pub fn forward(
		&self,
		q: AutogradTensor,
		k: AutogradTensor,
		v: AutogradTensor,
	) -> Result<AutogradTensor, ErrPack<TensorOpError>> {
		let (q, q_backward) = q.into_parts();
		let (k, k_backward) = k.into_parts();
		let (v, v_backward) = v.into_parts();
		let o = Self::alloc_output(&q, &k, &v)?;

		let (q_batch, q_map) = q.map().nd_split::<3>()?;
		let (k_batch, k_map) = k.map().nd_split::<3>()?;
		let (v_batch, v_map) = v.map().nd_split::<3>()?;
		let (o_batch, o_map) = o.map().nd_split::<3>()?;

		let q_count = q_map.dims[0].size;
		let k_count = k_map.dims[0].size;
		let v_count = v_map.dims[0].size;
		let o_count = o_map.dims[0].size;

		let q_width = q_map.dims[2].size;
		let k_width = k_map.dims[2].size;
		let v_width = v_map.dims[2].size;
		let o_width = o_map.dims[2].size;

		let q_heads = q_map.dims[1].size;
		let k_heads = k_map.dims[1].size;
		let v_heads = v_map.dims[1].size;
		let o_heads = o_map.dims[1].size;
		let group_shift = q_heads.trailing_zeros().wrapping_sub(k_heads.trailing_zeros());
		if k_heads > q_heads
			|| k_heads != v_heads
			|| (k_heads << group_shift) != q_heads
			|| k_heads == 0
			|| q_width != k_width
			|| v_count != k_count
			|| o_count != q_count
			|| o_heads != q_heads
			|| o_width != v_width
		{
			cold_path();
			return Err(TensorOpError::shape_mismatch());
		}

		let attention_args = AttentionArgs {
			q_count,
			head_count: q_heads,
			q_width,
			q_offset: q_map.offset,
			q_item_stride: q_map.dims[0].stride,
			q_head_stride: q_map.dims[1].stride,
			q: q.buf().device_data(),

			k_count,
			group_shift: group_shift as usize,
			// k_width == q_width
			k_offset: k_map.offset,
			k_item_stride: k_map.dims[0].stride,
			k_head_stride: k_map.dims[1].stride,
			k: k.buf().device_data(),

			// v_count == k_count
			// v_head_count == head_count >> group_shift
			v_width,
			v_offset: v_map.offset,
			v_item_stride: v_map.dims[0].stride,
			v_head_stride: v_map.dims[1].stride,
			v: v.buf().device_data(),

			// o_count == q_count
			// o_head_count == head_count
			// o_width == v_width
			o_offset: o_map.offset,
			o_head_stride: o_map.dims[1].stride,
			o_item_stride: o_map.dims[0].stride,
			o: o.buf().device_data(),
		};

		// TODO:
		// - merge batch dims (0..-3) into one batch dim
		// - add batch loop
		// - borrow tensors

		let mut inp_fail = UnsafeBorrowFailFlag::new();
		q.borrow();
		o.vmt().attention(&attention_args)?;
		Ok(())
	}
}

impl Fragment for Attention {
	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn randomize(&mut self, _rng: &mut Rng) -> Result<(), ErrPack<TensorOpError>> {
		// no parameters to randomize
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct AttentionBackwardFn_Precise {
	q_backward: Box<dyn BackwardFn>,
	k_backward: Box<dyn BackwardFn>,
	v_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for AttentionBackwardFn_Precise {
	fn run(
		self: Box<Self>,
		_d_out: Tensor,
		_queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("Implement Attention backward pass");
	}
}

//--------------------------------------------------------------------------------------------------
