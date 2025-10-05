//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::Arc;

use crate::ErrPack;
use crate::tensor::device::DeviceBase;
use crate::tensor::device::dtype::common_dtype;
use crate::tensor::device::kernel::runner::KernelData;
use crate::tensor::{DType, Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

#[macro_export]
macro_rules! custom_kernel {
	(
		$internal_dtype:expr,
		[ $($tensor_id:ident : $tensor_expr:expr),* $(,)? ],
		( $($scalar_id:ident : $scalar_expr:expr),* $(,)? ),
		$body:expr
	) => {{
		$crate::tensor::device::kernel::expr::KernelCall::new(
			[ $($tensor_expr),* ],
			[ $($scalar_expr),* ],
			{
				$(
					let $tensor_id = $crate::tensor::device::kernel::expr::Expr(
						$crate::tensor::device::kernel::expr::TensorArg::<${index()}>
					);
				)*
				$(
					let $scalar_id = $crate::tensor::device::kernel::expr::Expr(
						$crate::tensor::device::kernel::expr::ScalarArg::<${index()}>
					);
				)*
				$body
			},
			$internal_dtype
		)
	}};
}

//--------------------------------------------------------------------------------------------------

pub trait EvaluatesToTensor {
	/// Calculate the result of the operation represented by `self`
	/// and save it into the `to` tensor.
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>>;
}

impl EvaluatesToTensor for f64 {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let internal_dtype = to.dtype();
		to.assign(custom_kernel!(
			internal_dtype,
			[], (VALUE: self), {
				VALUE
			}
		))
	}
}

impl EvaluatesToTensor for &Tensor {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let internal_dtype = common_dtype(self.dtype(), to.dtype())?;
		to.assign(custom_kernel!(
			internal_dtype,
			[src: self], (), {
				src
			}
		))
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ExprDiscriminant {
	ElemwiseTensorArg = 1,
	ReduceTensorArg,
	ScalarArg,

	SumExpr,
	MaxExpr,

	NegExpr,
	ExpExpr,
	LnExpr,
	AbsExpr,
	SqrtExpr,
	RecipExpr,

	AddExpr,
	SubExpr,
	MulExpr,
}

//--------------------------------------------------------------------------------------------------

pub enum DynExpr {
	ElemwiseTensorArg(usize),
	ReduceTensorArg(usize),
	ScalarArg(usize),

	SumExpr(Arc<DynExpr>),
	MaxExpr(Arc<DynExpr>),

	NegExpr(Arc<DynExpr>),
	ExpExpr(Arc<DynExpr>),
	LnExpr(Arc<DynExpr>),
	AbsExpr(Arc<DynExpr>),
	SqrtExpr(Arc<DynExpr>),
	RecipExpr(Arc<DynExpr>),

	AddExpr(Arc<DynExpr>, Arc<DynExpr>),
	SubExpr(Arc<DynExpr>, Arc<DynExpr>),
	MulExpr(Arc<DynExpr>, Arc<DynExpr>),
}

//--------------------------------------------------------------------------------------------------

pub type KernelKeyType = u64;
pub const KEY_TYPE_SIZE: usize = std::mem::size_of::<KernelKeyType>();

union KeyUnion<const KEY_LEN: usize>
where
	[(); KEY_TYPE_SIZE * KEY_LEN]:,
{
	id: [u8; KEY_TYPE_SIZE * KEY_LEN],
	key: [KernelKeyType; KEY_LEN],
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct InputMasks {
	pub elemwise: u64,
	pub reduce: u64,
	pub scalar: u64,
}

#[derive(Clone, Copy)]
pub struct InputCounts {
	pub elemwise: usize,
	pub reduce: usize,
	pub scalar: usize,
}

impl InputCounts {
	pub const fn new(masks: InputMasks) -> Self {
		Self {
			elemwise: masks.elemwise.count_ones() as usize,
			reduce: masks.reduce.count_ones() as usize,
			scalar: masks.scalar.count_ones() as usize,
		}
	}
}

#[const_trait]
pub trait ExprTrait {
	const MASKS: InputMasks;
	const COUNTS: InputCounts = InputCounts::new(Self::MASKS);

	const ELEMWISE_MASK: u64 = Self::MASKS.elemwise;
	const REDUCE_MASK: u64 = Self::MASKS.reduce;
	const SCALAR_MASK: u64 = Self::MASKS.scalar;

	const ELEMWISE_COUNT: usize = Self::COUNTS.elemwise;
	const REDUCE_COUNT: usize = Self::COUNTS.reduce;
	const SCALAR_COUNT: usize = Self::COUNTS.scalar;

	const REDUCE_OP_COUNT: usize;

	const DTYPE_CONFIG_LEN: usize = 2 + Self::ELEMWISE_COUNT + Self::REDUCE_COUNT;
	const EXPR_KEY_LEN: usize;

	// The total key consists of:
	// - dtype configuration (2 + ELEMWISE_COUNT + REDUCE_COUNT bytes)
	// - 1 byte separator with value 0
	// - padding to the next multiple of KEY_TYPE_SIZE with value 0
	// - expression key (EXPR_KEY_LEN bytes)
	// - padding to the next multiple of KEY_TYPE_SIZE with value 0
	// Note that both DTypeId and ExprDiscriminant start from 1, so 0 is never a valid value.
	const DTYPE_CONFIG_BYTES: usize = Self::DTYPE_CONFIG_WORDS * KEY_TYPE_SIZE;
	const EXPR_KEY_BYTES: usize = Self::EXPR_KEY_WORDS * KEY_TYPE_SIZE;

	const DTYPE_CONFIG_WORDS: usize =
		(Self::DTYPE_CONFIG_LEN + 1).next_multiple_of(KEY_TYPE_SIZE) / KEY_TYPE_SIZE;
	const EXPR_KEY_WORDS: usize =
		Self::EXPR_KEY_LEN.next_multiple_of(KEY_TYPE_SIZE) / KEY_TYPE_SIZE;

	const KEY_WORDS: usize = Self::DTYPE_CONFIG_WORDS + Self::EXPR_KEY_WORDS;

	fn set_key(masks: InputMasks, reduce: bool, id: &mut [u8], i: usize) -> usize;

	#[allow(clippy::indexing_slicing)]
	fn key() -> [u64; Self::KEY_WORDS]
	where
		[(); KEY_TYPE_SIZE * Self::KEY_WORDS]:,
	{
		let mut u = KeyUnion { key: [0; Self::KEY_WORDS] };
		let bytes = unsafe { &mut u.id };
		let id = &mut bytes[Self::DTYPE_CONFIG_BYTES..];
		let len = Self::set_key(Self::MASKS, false, id, 0);
		assert!(len == Self::EXPR_KEY_LEN);
		unsafe { u.key }
	}
}

pub trait ExprToDyn {
	fn to_dyn_internal(&self, e_mask: u64, r_mask: u64, s_mask: u64, reduce: bool) -> Arc<DynExpr>;
	fn to_dyn(&self) -> Arc<DynExpr>;
}

#[derive(Clone, Copy)]
pub struct TensorArg<const Idx: usize>;
#[derive(Clone, Copy)]
pub struct ScalarArg<const Idx: usize>;

#[derive(Clone, Copy)]
pub struct SumExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct MaxExpr<A: const ExprTrait + ExprToDyn>(pub A);

#[derive(Clone, Copy)]
pub struct NegExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct ExpExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct LnExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct AbsExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct SqrtExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct RecipExpr<A: const ExprTrait + ExprToDyn>(pub A);

#[derive(Clone, Copy)]
pub struct AddExpr<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn>(pub A, pub B);
#[derive(Clone, Copy)]
pub struct SubExpr<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn>(pub A, pub B);
#[derive(Clone, Copy)]
pub struct MulExpr<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn>(pub A, pub B);

//--------------------------------------------------------------------------------------------------

impl<const Idx: usize> const ExprTrait for TensorArg<Idx> {
	const MASKS: InputMasks = InputMasks { elemwise: 1 << Idx, reduce: 0, scalar: 0 };

	const REDUCE_OP_COUNT: usize = 0;

	const EXPR_KEY_LEN: usize = 2;

	fn set_key(masks: InputMasks, reduce: bool, id: &mut [u8], i: usize) -> usize {
		let bit = 1_u64 << Idx;
		if reduce {
			let idx = (masks.reduce & (bit - 1)).count_ones() as usize;
			id[i + 0] = ExprDiscriminant::ReduceTensorArg as u8;
			id[i + 1] = idx as u8;
		} else {
			let idx = (masks.elemwise & (bit - 1)).count_ones() as usize;
			id[i + 0] = ExprDiscriminant::ElemwiseTensorArg as u8;
			id[i + 1] = idx as u8;
		}
		i + 2
	}
}

impl<const Idx: usize> ExprToDyn for TensorArg<Idx> {
	fn to_dyn_internal(&self, em: u64, rm: u64, _sm: u64, reduce: bool) -> Arc<DynExpr> {
		let bit = 1_u64 << Idx;
		if reduce {
			let idx = (rm & (bit - 1)).count_ones() as usize;
			Arc::new(DynExpr::ReduceTensorArg(idx))
		} else {
			let idx = (em & (bit - 1)).count_ones() as usize;
			Arc::new(DynExpr::ElemwiseTensorArg(idx))
		}
	}
	fn to_dyn(&self) -> Arc<DynExpr> {
		self.to_dyn_internal(Self::ELEMWISE_MASK, Self::REDUCE_MASK, Self::SCALAR_MASK, false)
	}
}

impl<const Idx: usize> const ExprTrait for ScalarArg<Idx> {
	const MASKS: InputMasks = InputMasks { elemwise: 0, reduce: 0, scalar: 1 << Idx };

	const REDUCE_OP_COUNT: usize = 0;

	const EXPR_KEY_LEN: usize = 2;

	fn set_key(masks: InputMasks, _reduce: bool, id: &mut [u8], i: usize) -> usize {
		let bit = 1_u64 << Idx;
		let idx = (masks.scalar & (bit - 1)).count_ones() as usize;
		id[i + 0] = ExprDiscriminant::ScalarArg as u8;
		id[i + 1] = idx as u8;
		i + 2
	}
}

impl<const Idx: usize> ExprToDyn for ScalarArg<Idx> {
	fn to_dyn_internal(
		&self,
		_e_mask: u64,
		_r_mask: u64,
		s_mask: u64,
		_reduce: bool,
	) -> Arc<DynExpr> {
		let bit = 1_u64 << Idx;
		let idx = (s_mask & (bit - 1)).count_ones() as usize;
		Arc::new(DynExpr::ScalarArg(idx))
	}
	fn to_dyn(&self) -> Arc<DynExpr> {
		self.to_dyn_internal(Self::ELEMWISE_MASK, Self::REDUCE_MASK, Self::SCALAR_MASK, false)
	}
}

macro_rules! impl_expr_reduce {
	($name:ident) => {
		impl<A: const ExprTrait + ExprToDyn> const ExprTrait for $name<A> {
			const MASKS: InputMasks = InputMasks {
				elemwise: 0,
				reduce: A::MASKS.reduce | A::MASKS.elemwise,
				scalar: A::MASKS.scalar,
			};

			const REDUCE_OP_COUNT: usize = 1 + A::REDUCE_OP_COUNT;

			const EXPR_KEY_LEN: usize = 1 + A::EXPR_KEY_LEN;

			fn set_key(masks: InputMasks, _reduce: bool, id: &mut [u8], i: usize) -> usize {
				id[i] = ExprDiscriminant::$name as u8;
				A::set_key(masks, true, id, i + 1)
			}
		}

		impl<A: const ExprTrait + ExprToDyn> ExprToDyn for $name<A> {
			fn to_dyn_internal(
				&self,
				e_mask: u64,
				r_mask: u64,
				s_mask: u64,
				reduce: bool,
			) -> Arc<DynExpr> {
				assert!(!reduce);
				Arc::new(DynExpr::$name(self.0.to_dyn_internal(e_mask, r_mask, s_mask, true)))
			}
			fn to_dyn(&self) -> Arc<DynExpr> {
				self.to_dyn_internal(
					Self::ELEMWISE_MASK,
					Self::REDUCE_MASK,
					Self::SCALAR_MASK,
					false,
				)
			}
		}
	};
}

macro_rules! impl_expr_unary {
	($name:ident) => {
		impl<A: const ExprTrait + ExprToDyn> const ExprTrait for $name<A> {
			const MASKS: InputMasks = A::MASKS;

			const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT;

			const EXPR_KEY_LEN: usize = 1 + A::EXPR_KEY_LEN;

			fn set_key(masks: InputMasks, reduce: bool, id: &mut [u8], i: usize) -> usize {
				id[i + 0] = ExprDiscriminant::$name as u8;
				A::set_key(masks, reduce, id, i + 1)
			}
		}

		impl<A: const ExprTrait + ExprToDyn> ExprToDyn for $name<A> {
			fn to_dyn_internal(
				&self,
				e_mask: u64,
				r_mask: u64,
				s_mask: u64,
				reduce: bool,
			) -> Arc<DynExpr> {
				Arc::new(DynExpr::$name(self.0.to_dyn_internal(e_mask, r_mask, s_mask, reduce)))
			}
			fn to_dyn(&self) -> Arc<DynExpr> {
				self.to_dyn_internal(
					Self::ELEMWISE_MASK,
					Self::REDUCE_MASK,
					Self::SCALAR_MASK,
					false,
				)
			}
		}
	};
}

macro_rules! impl_expr_binary {
	($name:ident, $commutative:expr) => {
		impl<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn> const ExprTrait
			for $name<A, B>
		{
			const MASKS: InputMasks = InputMasks {
				elemwise: A::MASKS.elemwise | B::MASKS.elemwise,
				reduce: A::MASKS.reduce | B::MASKS.reduce,
				scalar: A::MASKS.scalar | B::MASKS.scalar,
			};

			const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT + B::REDUCE_OP_COUNT;

			const EXPR_KEY_LEN: usize = 1 + A::EXPR_KEY_LEN + B::EXPR_KEY_LEN;

			fn set_key(masks: InputMasks, reduce: bool, id: &mut [u8], i: usize) -> usize {
				id[i + 0] = ExprDiscriminant::$name as u8;
				let begin = i + 1;
				let mid = A::set_key(masks, reduce, id, begin);
				assert!(mid == begin + A::EXPR_KEY_LEN);
				let end = B::set_key(masks, reduce, id, mid);
				assert!(end == mid + B::EXPR_KEY_LEN);
				if !($commutative) {
					return end;
				}

				if A::EXPR_KEY_LEN <= B::EXPR_KEY_LEN {
					if A::EXPR_KEY_LEN == B::EXPR_KEY_LEN {
						let mut swap = false;
						let mut i = 0;
						while i < A::EXPR_KEY_LEN {
							let a = id[begin + i];
							let b = id[mid + i];
							if a != b {
								swap = a > b;
								break;
							}
							i += 1;
						}
						if swap {
							let mut i = 0;
							while i < A::EXPR_KEY_LEN {
								let tmp = id[begin + i];
								id[begin + i] = id[mid + i];
								id[mid + i] = tmp;
								i += 1;
							}
						}
					}
					end
				} else {
					let mid = B::set_key(masks, reduce, id, begin);
					assert!(mid == begin + B::EXPR_KEY_LEN);
					let end = A::set_key(masks, reduce, id, mid);
					assert!(end == mid + A::EXPR_KEY_LEN);
					end
				}
			}
		}

		impl<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn> ExprToDyn
			for $name<A, B>
		{
			fn to_dyn_internal(
				&self,
				e_mask: u64,
				r_mask: u64,
				s_mask: u64,
				reduce: bool,
			) -> Arc<DynExpr> {
				let a = self.0.to_dyn_internal(e_mask, r_mask, s_mask, reduce);
				let b = self.1.to_dyn_internal(e_mask, r_mask, s_mask, reduce);
				Arc::new(DynExpr::$name(a, b))
			}
			fn to_dyn(&self) -> Arc<DynExpr> {
				self.to_dyn_internal(
					Self::ELEMWISE_MASK,
					Self::REDUCE_MASK,
					Self::SCALAR_MASK,
					false,
				)
			}
		}
	};
}

impl_expr_reduce!(SumExpr);
impl_expr_reduce!(MaxExpr);

impl_expr_unary!(NegExpr);
impl_expr_unary!(ExpExpr);
impl_expr_unary!(LnExpr);
impl_expr_unary!(AbsExpr);
impl_expr_unary!(SqrtExpr);
impl_expr_unary!(RecipExpr);

impl_expr_binary!(AddExpr, true);
impl_expr_binary!(SubExpr, false);
impl_expr_binary!(MulExpr, true);

//--------------------------------------------------------------------------------------------------

pub struct KernelCall<'a, E: const ExprTrait + ExprToDyn>
where
	[(); E::ELEMWISE_COUNT]:,
	[(); E::REDUCE_COUNT]:,
	[(); E::SCALAR_COUNT]:,
{
	elem_args: [&'a Tensor; E::ELEMWISE_COUNT],
	reduce_args: [&'a Tensor; E::REDUCE_COUNT],
	scalar_args: [f64; E::SCALAR_COUNT],
	expr: E,
	internal_dtype: DType,
}

impl<'a, E: const ExprTrait + ExprToDyn + Copy> KernelCall<'a, E>
where
	[(); E::ELEMWISE_COUNT]:,
	[(); E::REDUCE_COUNT]:,
	[(); E::SCALAR_COUNT]:,
{
	const fn mask_to_indexes<const N: usize>(mut mask: u64) -> [usize; N] {
		assert!(mask.count_ones() as usize == N);
		let mut indexes = [0_usize; N];
		let mut i = 0;
		while i < N {
			let idx = mask.trailing_zeros() as usize;
			indexes[i] = idx;
			mask &= !(1 << idx);
			i += 1;
		}
		indexes
	}

	#[inline]
	pub fn new<const TC: usize, const SC: usize>(
		tensors: [&'a Tensor; TC],
		scalars: [f64; SC],
		expr: Expr<E>,
		internal_dtype: DType,
	) -> Self {
		let elem_arg_indexes: [usize; E::ELEMWISE_COUNT] =
			const { Self::mask_to_indexes(E::ELEMWISE_MASK) };
		let reduce_arg_indexes: [usize; E::REDUCE_COUNT] =
			const { Self::mask_to_indexes(E::REDUCE_MASK) };
		let scalar_arg_indexes: [usize; E::SCALAR_COUNT] =
			const { Self::mask_to_indexes(E::SCALAR_MASK) };
		Self {
			elem_args: std::array::from_fn(|i| tensors[elem_arg_indexes[i]]),
			reduce_args: std::array::from_fn(|i| tensors[reduce_arg_indexes[i]]),
			scalar_args: std::array::from_fn(|i| scalars[scalar_arg_indexes[i]]),
			expr: expr.0,
			internal_dtype,
		}
	}

	pub fn call(
		&self,
		output: &Tensor,
		elem_args: [&Tensor; E::ELEMWISE_COUNT],
		reduce_args: [&Tensor; E::REDUCE_COUNT],
		scalar_args: [f64; E::SCALAR_COUNT],
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); E::KEY_WORDS]:,
		[(); KEY_TYPE_SIZE * E::KEY_WORDS]:,
		[(); KernelData::dtype_config_words(E::ELEMWISE_COUNT, E::REDUCE_COUNT)]:,
		[(); E::DTYPE_CONFIG_WORDS
			- KernelData::dtype_config_words(E::ELEMWISE_COUNT, E::REDUCE_COUNT)]:,
		[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:,
	{
		let mut key = const { E::key() };
		let dtype_config =
			KernelData::new_dtype_config(internal_dtype, output, elem_args, reduce_args);
		key[..dtype_config.len()].copy_from_slice(&dtype_config);

		__run_kernel(&self.expr, &key, output, elem_args, reduce_args, scalar_args, internal_dtype)
	}
}

#[inline(never)]
#[allow(clippy::indexing_slicing)]
#[allow(clippy::too_many_lines)]
fn __run_kernel<const E: usize, const R: usize, const C: usize>(
	expr: &dyn ExprToDyn,
	key: &[u64],
	output: &Tensor,
	elem_args: [&Tensor; E],
	reduce_args: [&Tensor; R],
	scalar_args: [f64; C],
	internal_dtype: DType,
) -> Result<(), ErrPack<TensorOpError>>
where
	[(); 1 + E + R]:,
{
	debug_assert!(kernel_data.elemwise_count == E);
	debug_assert!(kernel_data.reduce_count == R);
	debug_assert!(kernel_data.scalar_count == C);

	let dtype_bytes = output.buf().dtype().bytes();
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
			return Err(TensorOpError::cannot_broadcast_output());
		}
		if reduce_args_top_dim.iter().any(|vec| vec.stride != 1) {
			cold_path();
			return Err(NotContiguousError.into());
		}
	}

	let all_dims_tmp = crate::util::array::concat_arrays([output_batch_dims], elem_args_batch_dims);
	let all_dims = crate::util::array::concat_arrays(all_dims_tmp, reduce_args_batch_dims);

	let merged = DimMerger::merge::<2>(all_dims)?;

	let reduce_args_top_dim = DimMerger::merge_single_dim(reduce_args_top_dim)?;
	let reduce_inp: [KernelReduceArg; R] = std::array::from_fn(|i| {
		let arg = reduce_args[i];
		KernelReduceArg {
			stride_bytes: [
				merged[0].strides[1 + E + i] * dtype_bytes,
				merged[1].strides[1 + E + i] * dtype_bytes,
				reduce_args_top_dim.strides[i] * dtype_bytes,
			],
			offset_bytes: arg.map().offset * dtype_bytes,
			buf: arg.buf().memory(),
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
			buf: arg.buf().memory(),
		}
	});

	if merged.iter().any(|m| m.get(0).is_broadcasted()) {
		cold_path();
		return Err(TensorOpError::cannot_broadcast_output());
	}

	let out = KernelOutput {
		size: [merged[0].size, merged[1].size],
		stride_bytes: [
			merged[0].strides[0] * dtype_bytes, //
			merged[1].strides[0] * dtype_bytes,
		],
		offset_bytes: output.map().offset * dtype_bytes,
		buf: output.buf().memory(),
		reduction_size: reduce_args_top_dim.size,
	};

	let dtype_config = KernelData::new_dtype_config(internal_dtype, output, elem_args, reduce_args);

	unsafe {
		let mut inp_fail = UnsafeBorrowFailFlag::new();
		let reduce_borrows: [BorrowGuard<DeviceBuffer>; R] =
			std::array::from_fn(|i| reduce_args[i].buf().unsafe_borrow(&mut inp_fail));
		let elem_borrows: [Option<BorrowGuard<DeviceBuffer>>; E] = std::array::from_fn(|i| {
			let arg = &elem_args[i];
			let same_as_output = std::ptr::eq(arg.buf().as_ref(), output.buf().as_ref())
				&& likely(
					inp[i].offset_bytes == out.offset_bytes
						&& inp[i].stride_bytes == out.stride_bytes,
				);
			if same_as_output { None } else { Some(arg.buf().unsafe_borrow(&mut inp_fail)) }
		});

		let mut out_fail = UnsafeBorrowMutFailFlag::new();
		let out_borrow = output.buf().unsafe_borrow_mut(&mut out_fail);

		inp_fail.check()?;
		out_fail.check()?;

		// TODO - ensure_safe
		// TODO - ensure all on same device
		// TODO - other things may need to be checked before running the kernel

		output.device().run_kernel(
			kernel_data,
			&out,
			inp.as_ptr(),
			reduce_inp.as_ptr(),
			scalar_args.as_ptr(),
			dtype_config.as_ptr(),
		)?;

		std::mem::drop(out_borrow);
		std::mem::drop(elem_borrows);
		std::mem::drop(reduce_borrows);
	}
	Ok(())
}

impl<'a, E: const ExprTrait + ExprToDyn> EvaluatesToTensor for KernelCall<'a, E>
where
	[(); 1 - E::REDUCE_OP_COUNT]:,
	[(); E::ELEMWISE_COUNT]:,
	[(); E::REDUCE_COUNT]:,
	[(); E::SCALAR_COUNT]:,
	[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:,
	[(); E::KEY_WORDS]:,
	[(); KEY_TYPE_SIZE * E::KEY_WORDS]:,
{
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		DeviceBase::from_device(to.device()).kernel_runner.run(
			to,
			self.elem_args,
			self.reduce_args,
			self.scalar_args,
			self.internal_dtype,
		)
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Expr<E: const ExprTrait + ExprToDyn + Copy>(pub E);

impl<E: const ExprTrait + ExprToDyn + Copy> Expr<E> {
	/// Computes the sum along the last dimension.
	///
	/// This is equivalent to PyTorch's `tensor.sum(dim=-1, keepdim=True)`.
	pub const fn sum(self) -> Expr<SumExpr<E>> {
		Expr(SumExpr(self.0))
	}

	/// Computes the maximum along the last dimension.
	///
	/// This is equivalent to PyTorch's `tensor.max(dim=-1, keepdim=True)`.
	pub const fn max(self) -> Expr<MaxExpr<E>> {
		Expr(MaxExpr(self.0))
	}

	pub const fn exp(self) -> Expr<ExpExpr<E>> {
		Expr(ExpExpr(self.0))
	}

	pub const fn ln(self) -> Expr<LnExpr<E>> {
		Expr(LnExpr(self.0))
	}

	pub const fn abs(self) -> Expr<AbsExpr<E>> {
		Expr(AbsExpr(self.0))
	}

	pub const fn sqrt(self) -> Expr<SqrtExpr<E>> {
		Expr(SqrtExpr(self.0))
	}

	pub const fn recip(self) -> Expr<RecipExpr<E>> {
		Expr(RecipExpr(self.0))
	}
}

impl<E: const ExprTrait + ExprToDyn + Copy> std::ops::Neg for Expr<E> {
	type Output = Expr<NegExpr<E>>;

	fn neg(self) -> Expr<NegExpr<E>> {
		Expr(NegExpr(self.0))
	}
}

impl<A: const ExprTrait + ExprToDyn + Copy, E: const ExprTrait + ExprToDyn + Copy>
	std::ops::Add<Expr<E>> for Expr<A>
{
	type Output = Expr<AddExpr<A, E>>;

	fn add(self, rhs: Expr<E>) -> Expr<AddExpr<A, E>> {
		Expr(AddExpr(self.0, rhs.0))
	}
}

impl<A: const ExprTrait + ExprToDyn + Copy, E: const ExprTrait + ExprToDyn + Copy>
	std::ops::Sub<Expr<E>> for Expr<A>
{
	type Output = Expr<SubExpr<A, E>>;

	fn sub(self, rhs: Expr<E>) -> Expr<SubExpr<A, E>> {
		Expr(SubExpr(self.0, rhs.0))
	}
}

impl<A: const ExprTrait + ExprToDyn + Copy, E: const ExprTrait + ExprToDyn + Copy>
	std::ops::Mul<Expr<E>> for Expr<A>
{
	type Output = Expr<MulExpr<A, E>>;

	fn mul(self, rhs: Expr<E>) -> Expr<MulExpr<A, E>> {
		Expr(MulExpr(self.0, rhs.0))
	}
}

//--------------------------------------------------------------------------------------------------
