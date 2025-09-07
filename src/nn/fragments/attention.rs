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
use crate::tensor::generic::map::SizeAndStride;
use crate::tensor::generic::map::dd::{DimVecBuilder, INLINE_DIMS};
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
		for i in 0..ndim - 2 {
			let q_size = q.size(i).unwrap_or(1);
			let k_size = k.size(i).unwrap_or(1);
			let v_size = v.size(i).unwrap_or(1);
			shape[i] = q_size.max(k_size).max(v_size);
		}
		shape[ndim - 2] = q.size(-2).unwrap();
		shape[ndim - 1] = v.size(-1).unwrap();

		q.new_empty(shape, q.dtype())
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

		// TODO:
		// - merge dims 0..-3 into one batch dim
		// - prepare ND tensors
		// - call attention kernel

		let mut inp_fail = UnsafeBorrowFailFlag::new();
		q.borrow();
		o.vmt().attention(&mut o, &q, &k, &v)?;
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
