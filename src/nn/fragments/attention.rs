//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::hint::{cold_path, likely};
use std::ptr::NonNull;
use std::rc::Rc;

use smallvec::{SmallVec, smallvec};

use crate::autograd::{AutogradTensor, BackwardFn};
use crate::nn::Param;
use crate::nn::fragments::Fragment;
use crate::rng::Rng;
use crate::tensor::device::buffer::AttentionArgs;
use crate::tensor::dim_merger::DimMerger;
use crate::tensor::generic::map::dd::INLINE_DIMS;
use crate::tensor::{Tensor, TensorOpError};
use crate::util::mycell::{UnsafeBorrowFailFlag, UnsafeBorrowMutFailFlag};
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
	#[allow(clippy::indexing_slicing)]
	#[allow(clippy::needless_range_loop)]
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
		shape[ndim - 3] = q.size(-3).unwrap();
		shape[ndim - 2] = q.size(-2).unwrap();
		shape[ndim - 1] = v.size(-1).unwrap();
		let mut can_reuse = shape[ndim - 1] == q.size(-1).unwrap();
		for i in 0..ndim - 3 {
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

		q.ensure_safe()?;
		k.ensure_safe()?;
		v.ensure_safe()?;

		let (q_batch, q_map) = q.map().nd_split::<3>()?;
		let (k_batch, k_map) = k.map().nd_split::<3>()?;
		let (v_batch, v_map) = v.map().nd_split::<3>()?;
		let (o_batch, o_map) = o.map().nd_split::<3>()?;

		let [q_count, q_heads, q_width] = q_map.dims;
		let [k_count, k_heads, k_width] = k_map.dims;
		let [v_count, v_heads, v_width] = v_map.dims;
		let [o_count, o_heads, o_width] = o_map.dims;

		let group_shift = q_heads.size.trailing_zeros().wrapping_sub(k_heads.size.trailing_zeros());
		if k_heads.size > q_heads.size
			|| k_heads.size != v_heads.size
			|| (k_heads.size << group_shift) != q_heads.size
			|| k_heads.size == 0
			|| q_width.size != k_width.size
			|| v_count.size != k_count.size
			|| o_count.size != q_count.size
			|| o_heads.size != q_heads.size
			|| o_width.size != v_width.size
		{
			cold_path();
			return Err(TensorOpError::shape_mismatch());
		}
		if !q_width.is_contiguous()
			|| !k_width.is_contiguous()
			|| !v_width.is_contiguous()
			|| !o_width.is_contiguous()
		{
			cold_path();
			return Err(TensorOpError::not_contiguous());
		}
		if q.dtype() != k.dtype() || q.dtype() != v.dtype() || q.dtype() != o.dtype() {
			cold_path();
			return Err(TensorOpError::dtype_mismatch());
		}

		let mut args = AttentionArgs {
			q_count: q_count.size,
			head_count: q_heads.size,
			q_width: q_width.size,
			q_offset: q_map.offset,
			q_item_stride: q_count.stride,
			q_head_stride: q_heads.stride,
			q: q.buf().device_data(),

			k_count: k_count.size,
			group_shift: group_shift as usize,
			// k_width == q_width
			k_offset: k_map.offset,
			k_item_stride: k_count.stride,
			k_head_stride: k_heads.stride,
			k: k.buf().device_data(),

			// v_count == k_count
			// v_head_count == head_count >> group_shift
			v_width: v_width.size,
			v_offset: v_map.offset,
			v_item_stride: v_count.stride,
			v_head_stride: v_heads.stride,
			v: v.buf().device_data(),

			// o_count == q_count
			// o_head_count == head_count
			// o_width == v_width
			o_offset: o_map.offset,
			o_item_stride: o_count.stride,
			o_head_stride: o_heads.stride,
			o: o.buf().device_data(),
		};

		{
			let mut inp_fail = UnsafeBorrowFailFlag::new();
			let _q_borrow = unsafe {
				let same_as_output = std::ptr::eq(q.buf().as_ref(), o.buf().as_ref())
					&& likely(
						args.q_offset == args.o_offset
							&& (args.q_item_stride == args.o_item_stride || args.q_count == 1)
							&& (args.q_head_stride == args.o_head_stride || args.head_count == 1),
					);
				if same_as_output { None } else { Some(q.buf().unsafe_borrow(&mut inp_fail)) }
			};
			let _k_borrow = unsafe { k.buf().unsafe_borrow(&mut inp_fail) };
			let _v_borrow = unsafe { v.buf().unsafe_borrow(&mut inp_fail) };
			inp_fail.check()?;
			let mut out_fail = UnsafeBorrowMutFailFlag::new();
			let _o_borrow = unsafe { o.buf().unsafe_borrow_mut(&mut out_fail) };
			out_fail.check()?;

			let m = DimMerger::merge::<1>([q_batch, k_batch, v_batch, o_batch])?;

			let vmt = o.vmt();
			for _ in 0..m[0].size {
				unsafe { (vmt.attention)(NonNull::from_ref(vmt), &args) }?;
				args.q_offset += m[0].strides[0];
				args.k_offset += m[0].strides[1];
				args.v_offset += m[0].strides[2];
				args.o_offset += m[0].strides[3];
			}
		}

		Ok(AutogradTensor::new(o, None)) // TODO: backward
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
