//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::{cold_path, likely};
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::sync::Arc;

use const_siphasher::sip::SipHasher13;
use hashbrown::HashTable;

use crate::ErrPack;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut, check_borrows};
use crate::tensor::device::executor::{KernelElemArg, KernelOutput, KernelReduceArg};
use crate::tensor::device::kernel::KernelData;
use crate::tensor::dim_merger::DimMerger;
use crate::tensor::generic::map::SizeAndStride;
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ExprDiscriminant {
	TensorArg,
	ScalarArg,

	DotExpr,

	SigmoidExpr,
	SwishExpr,
	SqrtExpr,
	RecipExpr,
	LnClampedExpr,
	AddExpr,
	MulExpr,

	Invalid,
}

//--------------------------------------------------------------------------------------------------

pub enum DynExpr {
	TensorArg(usize),
	ScalarArg(usize),

	DotExpr(Rc<DynExpr>, Rc<DynExpr>),

	SigmoidExpr(Rc<DynExpr>),
	SwishExpr(Rc<DynExpr>),
	SqrtExpr(Rc<DynExpr>),
	RecipExpr(Rc<DynExpr>, Rc<DynExpr>),
	LnClampedExpr(Rc<DynExpr>),
	AddExpr(Rc<DynExpr>, Rc<DynExpr>),
	MulExpr(Rc<DynExpr>, Rc<DynExpr>),
}

//--------------------------------------------------------------------------------------------------

#[const_trait]
pub trait Expr {
	const CONST: bool;
	const ELEMWISE_COUNT: usize;
	const REDUCE_COUNT: usize;
	const SCALAR_COUNT: usize;
	const KEY_LEN: usize;

	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize;
	fn elemwise_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize;
	fn reduce_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize;
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize;
}

pub trait ExprToDyn {
	fn to_dyn(self, e: &mut usize, r: &mut usize, s: &mut usize) -> Arc<DynExpr>;
}

pub struct TensorArg<'a>(pub &'a Tensor);
pub struct ScalarArg(pub f64);

pub struct SumExpr<A: Expr>(pub A);

pub struct SigmoidExpr<A: Expr>(pub A);
pub struct SwishExpr<A: Expr>(pub A);
pub struct SqrtExpr<A: Expr>(pub A);
pub struct RecipExpr<A: Expr, B: Expr>(pub A, pub B);
pub struct LnClampedExpr<A: Expr>(pub A);
pub struct AddExpr<A: Expr, B: Expr>(pub A, pub B);
pub struct MulExpr<A: Expr, B: Expr>(pub A, pub B);

//--------------------------------------------------------------------------------------------------

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<'a> const Expr for TensorArg<'a> {
	const CONST: bool = true;
	const ELEMWISE_COUNT: usize = 1;
	const REDUCE_COUNT: usize = 0;
	const SCALAR_COUNT: usize = 0;
	const KEY_LEN: usize = 1;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::TensorArg;
		i + 1
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		tensors[i].write(self.0);
		i + 1
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, _tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		i
	}

	#[inline(always)]
	fn scalars(&self, _scalars: &mut [f64], i: usize) -> usize {
		i
	}
}

impl<'a> ExprToDyn for TensorArg<'a> {
	fn to_dyn(self, e: &mut usize, r: &mut usize, s: &mut usize) -> Arc<DynExpr> {
		let result = Arc::new(DynExpr::TensorArg(*e));
		*e += 1;
		result
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl const Expr for ScalarArg {
	const CONST: bool = true;
	const ELEMWISE_COUNT: usize = 0;
	const REDUCE_COUNT: usize = 0;
	const SCALAR_COUNT: usize = 1;
	const KEY_LEN: usize = 1;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::ScalarArg;
		i + 1
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, _tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		i
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, _tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		i
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		scalars[i] = self.0;
		i + 1
	}
}

impl ExprToDyn for ScalarArg {
	fn to_dyn(self, e: &mut usize, r: &mut usize, s: &mut usize) -> Arc<DynExpr> {
		let result = Arc::new(DynExpr::ScalarArg(*s));
		*s += 1;
		result
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr> const Expr for SumExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = 0;
	const REDUCE_COUNT: usize = A::ELEMWISE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::DotExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, _tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		i
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr> const Expr for SigmoidExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::SigmoidExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		self.0.reduce_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr> const Expr for SwishExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::SwishExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		self.0.reduce_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr> const Expr for SqrtExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::SqrtExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		self.0.reduce_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr, B: const Expr> const Expr for RecipExpr<A, B> {
	const CONST: bool = A::CONST && B::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT + B::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN + B::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::RecipExpr;
		let m = A::key(id, i + 1);
		assert!(m == i + 1 + A::KEY_LEN);
		B::key(id, i + 1 + A::KEY_LEN)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		let m = self.0.elemwise_tensors(tensors, i);
		assert!(m == i + A::ELEMWISE_COUNT);
		self.1.elemwise_tensors(tensors, i + A::ELEMWISE_COUNT)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		let m = self.0.reduce_tensors(tensors, i);
		assert!(m == i + A::REDUCE_COUNT);
		self.1.reduce_tensors(tensors, i + A::REDUCE_COUNT)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		let m = self.0.scalars(scalars, i);
		assert!(m == i + A::SCALAR_COUNT);
		self.1.scalars(scalars, i + A::SCALAR_COUNT)
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr> const Expr for LnClampedExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::LnClampedExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		self.0.reduce_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr, B: const Expr> const Expr for AddExpr<A, B> {
	const CONST: bool = A::CONST && B::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT + B::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN + B::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::AddExpr;
		let m = A::key(id, i + 1);
		assert!(m == i + 1 + A::KEY_LEN);
		B::key(id, i + 1 + A::KEY_LEN)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		let m = self.0.elemwise_tensors(tensors, i);
		assert!(m == i + A::ELEMWISE_COUNT);
		self.1.elemwise_tensors(tensors, i + A::ELEMWISE_COUNT)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		let m = self.0.reduce_tensors(tensors, i);
		assert!(m == i + A::REDUCE_COUNT);
		self.1.reduce_tensors(tensors, i + A::REDUCE_COUNT)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		let m = self.0.scalars(scalars, i);
		assert!(m == i + A::SCALAR_COUNT);
		self.1.scalars(scalars, i + A::SCALAR_COUNT)
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr, B: const Expr> const Expr for MulExpr<A, B> {
	const CONST: bool = A::CONST && B::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT + B::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN + B::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::MulExpr;
		let m = A::key(id, i + 1);
		assert!(m == i + 1 + A::KEY_LEN);
		B::key(id, i + 1 + A::KEY_LEN)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		let m = self.0.elemwise_tensors(tensors, i);
		assert!(m == i + A::ELEMWISE_COUNT);
		self.1.elemwise_tensors(tensors, i + A::ELEMWISE_COUNT)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&'t self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize {
		let m = self.0.reduce_tensors(tensors, i);
		assert!(m == i + A::REDUCE_COUNT);
		self.1.reduce_tensors(tensors, i + A::REDUCE_COUNT)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		let m = self.0.scalars(scalars, i);
		assert!(m == i + A::SCALAR_COUNT);
		self.1.scalars(scalars, i + A::SCALAR_COUNT)
	}
}

//--------------------------------------------------------------------------------------------------

const KEY_BATCH: usize = std::mem::size_of::<u64>() / std::mem::size_of::<ExprDiscriminant>();

union KeyUnion<const N: usize>
where
	[(); N / KEY_BATCH]:,
{
	discriminants: [ExprDiscriminant; N],
	key: [u64; N / KEY_BATCH],
}

//--------------------------------------------------------------------------------------------------

struct CacheItem {
	hash: u64,
	value: Arc<KernelData>,
}

pub struct KernelRunner {
	cache: HashTable<CacheItem>,
}

impl KernelRunner {
	pub fn run<E: const Expr>(
		&self,
		output: &Tensor,
		expr: &E,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); E::ELEMWISE_COUNT]:,
		[(); E::REDUCE_COUNT]:,
		[(); E::SCALAR_COUNT]:,
		[(); E::KEY_LEN.next_multiple_of(KEY_BATCH)]:,
		[(); E::KEY_LEN.next_multiple_of(KEY_BATCH) / KEY_BATCH]:,
		[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:,
	{
		if !E::CONST {
			todo!("non-constant exprs not implemented yet");
		}

		let mut elem_args = [MaybeUninit::uninit(); E::ELEMWISE_COUNT];
		let cnt = expr.elemwise_tensors(&mut elem_args, 0);
		assert!(cnt == E::ELEMWISE_COUNT);
		let elem_args = unsafe { MaybeUninit::array_assume_init(elem_args) };

		let mut reduce_args = [MaybeUninit::uninit(); E::REDUCE_COUNT];
		let cnt = expr.reduce_tensors(&mut reduce_args, 0);
		assert!(cnt == E::REDUCE_COUNT);
		let reduce_args = unsafe { MaybeUninit::array_assume_init(reduce_args) };

		let mut scalar_args = [0_f64; E::SCALAR_COUNT];
		let cnt = expr.scalars(&mut scalar_args, 0);
		assert!(cnt == E::SCALAR_COUNT);
		let scalar_args = scalar_args;

		let (key, key_hash) = const { Self::get_key::<E>() };
		let kernel = if let Some(found) = self.find_kernel(&key, key_hash) {
			found
		} else {
			cold_path();
			self.add_kernel(key, key_hash, expr)?
		};
		Self::dispatch(kernel, output, elem_args, reduce_args, scalar_args)
	}

	#[allow(clippy::indexing_slicing)]
	const fn get_key<E: const Expr>()
	-> ([u64; E::KEY_LEN.next_multiple_of(KEY_BATCH) / KEY_BATCH], u64)
	where
		[(); E::KEY_LEN.next_multiple_of(KEY_BATCH) / KEY_BATCH]:,
	{
		let mut discriminants = [ExprDiscriminant::Invalid; E::KEY_LEN.next_multiple_of(KEY_BATCH)];
		let len = E::key(&mut discriminants, 0);
		assert!(len == E::KEY_LEN);

		let key_union = KeyUnion { discriminants };
		let key = unsafe { key_union.key };

		let mut key_hasher =
			SipHasher13::new_with_keys(3141_5926_5358_9793_u64, 2384_6264_3383_2795_u64);
		let mut i = 0;
		while i < key.len() {
			key_hasher.write_u64(key[i]);
			i += 1;
		}

		(key, key_hasher.finish())
	}

	#[inline(never)]
	fn find_kernel(&self, key: &[u64], key_hash: u64) -> Option<&KernelData> {
		self.cache
			.find(key_hash, |item| item.hash == key_hash && likely(item.value.key.as_ref() == key))
			.map(|item| item.value.as_ref())
	}

	fn add_kernel<E: const Expr>(
		&self,
		key: [u64; E::KEY_LEN.next_multiple_of(KEY_BATCH) / KEY_BATCH],
		key_hash: u64,
		expr: &E,
	) -> Result<&KernelData, ErrPack<TensorOpError>> {
		//
	}

	#[allow(clippy::indexing_slicing)]
	#[allow(clippy::too_many_lines)]
	fn dispatch<const E: usize, const R: usize, const C: usize>(
		kernel_data: &KernelData,
		output: &Tensor,
		elem_args: [&Tensor; E],
		reduce_args: [&Tensor; R],
		const_args: [f64; C],
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); 1 + E + R]:,
	{
		debug_assert!(kernel_data.elem_args.len() == E);
		debug_assert!(kernel_data.reduce_args.len() == R);
		debug_assert!(kernel_data.const_args.len() == C);

		let dtype_bytes = output.buf().dtype.bytes();
		debug_assert!(dtype_bytes > 0);

		let output_batch_dims: &[SizeAndStride];
		let elem_args_batch_dims: [&[SizeAndStride]; E];
		let reduce_args_batch_dims: [&[SizeAndStride]; R];
		let reduce_args_top_dim: [SizeAndStride; R];
		if R == 0 {
			reduce_args_top_dim = [SizeAndStride::default(); R];
			reduce_args_batch_dims = [&[]; R];

			output_batch_dims = output.map().dims.as_slice();
			elem_args_batch_dims = elem_args.map(|t| t.map().dims.as_slice());
		} else {
			let output_dims = output.map().dims.as_slice().split_last();
			let elem_args_dims = elem_args.try_map(|t| t.map().dims.as_slice().split_last());
			let reduce_args_dims = reduce_args.try_map(|t| t.map().dims.as_slice().split_last());

			let (Some(output_dims), Some(elem_dims), Some(reduce_dims)) =
				(output_dims, elem_args_dims, reduce_args_dims)
			else {
				cold_path();
				return Err(TensorOpError::missing_reduce_dimension());
			};

			let output_top_dim = output_dims.0;
			output_batch_dims = output_dims.1;
			let elem_args_top_dim = elem_dims.map(|(&top_dim, _)| top_dim);
			elem_args_batch_dims = elem_dims.map(|(_, batch_dim)| batch_dim);
			reduce_args_top_dim = reduce_dims.map(|(&top_dim, _)| top_dim);
			reduce_args_batch_dims = reduce_dims.map(|(_, batch_dim)| batch_dim);

			if output_top_dim.size != 1 || elem_args_top_dim.iter().any(|dim| dim.size != 1) {
				cold_path();
				// we would have to broadcast the result of the reduction
				// TODO - maybe use a different error
				return Err(TensorOpError::invalid_shape());
			}
			if reduce_args_top_dim.iter().any(|vec| vec.stride != 1) {
				cold_path();
				return Err(TensorOpError::not_contiguous());
			}
		}

		let all_dims_tmp =
			crate::util::array::concat_arrays([output_batch_dims], elem_args_batch_dims);
		let all_dims = crate::util::array::concat_arrays(all_dims_tmp, reduce_args_batch_dims);

		let merged = DimMerger::merge::<2>(all_dims)?;

		let reduce_inp: [KernelReduceArg; R] = std::array::from_fn(|i| {
			let arg = reduce_args[i];
			KernelReduceArg {
				reduction_size: reduce_args_top_dim[i].size,
				stride_bytes: [
					merged[0].strides[1 + E + i] * dtype_bytes,
					merged[1].strides[1 + E + i] * dtype_bytes,
				],
				offset_bytes: arg.map().offset * dtype_bytes,
				device_data: arg.buf().device_data,
			}
		});

		let inp: [KernelElemArg; E] = std::array::from_fn(|i| {
			let arg = elem_args[i];
			KernelElemArg {
				stride_bytes: [
					merged[0].strides[1 + i] * dtype_bytes,
					merged[1].strides[1 + i] * dtype_bytes,
				],
				offset_bytes: arg.map().offset * dtype_bytes,
				device_data: arg.buf().device_data,
			}
		});

		let out = [KernelOutput {
			size: [merged[0].size, merged[1].size],
			stride_bytes: [
				merged[0].strides[0] * dtype_bytes, //
				merged[1].strides[0] * dtype_bytes,
			],
			offset_bytes: output.map().offset * dtype_bytes,
			device_data: output.buf().device_data,
		}];

		unsafe {
			let mut c_fail = 0;
			let reduce_borrows: [DeviceBufferRef; R] = std::array::from_fn(|i| {
				let arg = &reduce_args[i];
				DeviceBufferRef::new_unsafe(arg.buf().as_ref(), &mut c_fail)
			});
			let elem_borrows: [Option<DeviceBufferRef>; E] = std::array::from_fn(|i| {
				let arg = &elem_args[i];
				let same_as_output = std::ptr::eq(arg.buf().as_ref(), output.buf().as_ref())
					&& likely(
						inp[i].offset_bytes == out[0].offset_bytes
							&& inp[i].stride_bytes == out[0].stride_bytes,
					);
				if same_as_output {
					None
				} else {
					Some(DeviceBufferRef::new_unsafe(arg.buf().as_ref(), &mut c_fail))
				}
			});

			let mut m_fail = 0;
			let out_borrow = DeviceBufferRefMut::new_unsafe(output.buf().as_ref(), &mut m_fail);

			check_borrows(c_fail, m_fail)?;

			// TODO - ensure_safe
			// TODO - ensure all on same device
			// TODO - other things may need to be checked before running the kernel

			output.executor().run_kernel(
				kernel_data,
				out.as_ptr(),
				inp.as_ptr(),
				reduce_inp.as_ptr(),
				const_args.as_ptr(),
			)?;

			std::mem::drop(out_borrow);
			std::mem::drop(elem_borrows);
			std::mem::drop(reduce_borrows);
		}
		Ok(())
	}
}
