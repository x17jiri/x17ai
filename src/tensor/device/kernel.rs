//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::{cold_path, likely};
use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::device::dtype::{DTypeId, common_dtype};
use crate::tensor::device::{DeviceBuffer, KernelArg, KernelOutput};
use crate::tensor::dim_merger::{DimMerger, DimsDontMatchError};
use crate::tensor::error::CannotBroadcastOutputError;
use crate::tensor::map::SizeAndStride;
use crate::tensor::{DType, Tensor, TensorOpError};
use crate::util::hasher::{HASH_WORD_SIZE, HashWord};
use crate::util::mycell::{BorrowGuard, UnsafeBorrowFailFlag};

//--------------------------------------------------------------------------------------------------

#[macro_export]
macro_rules! custom_kernel {
	(
		$internal_dtype:expr,
		[ $($tensor_id:ident : $tensor_expr:expr),* $(,)? ],
		( $($scalar_id:ident : $scalar_expr:expr),* $(,)? ),
		$body:expr
	) => {{
		$crate::tensor::device::kernel::KernelCall::new(
			[ $($tensor_expr),* ],
			[ $($scalar_expr),* ],
			{
				$(
					let $tensor_id = $crate::tensor::device::kernel::Expr(
						$crate::tensor::device::kernel::TensorArg::<${index()}>
					);
				)*
				$(
					let $scalar_id = $crate::tensor::device::kernel::Expr(
						$crate::tensor::device::kernel::ScalarArg::<${index()}>
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

pub struct DynExpr {
	pub kind: DynExprKind,

	/// This node is a reduction op (sum, max, ...),
	pub is_reduction: bool,

	/// Either this node itself is a reduction op (sum, max, ...),
	/// or it contains a reduction in its sub-expressions.
	pub has_reduction: bool,

	pub uses_elemwise_args: bool,
}

pub struct DynExprArg {
	pub kind: DynExprArgKind,
	pub index: usize,
}

pub enum DynExprArgKind {
	ElemwiseTensor,
	ReduceTensor,
	Scalar,
}

pub struct DynExprReduction {
	pub kind: DynExprReductionKind,
	pub expr: Rc<DynExpr>,
}

pub enum DynExprReductionKind {
	Sum,
	Max,
}

pub struct DynExprUnary {
	pub kind: DynExprUnaryKind,
	pub expr: Rc<DynExpr>,
}

pub enum DynExprUnaryKind {
	Neg,
	Exp,
	Ln,
	Abs,
	Sqrt,
	Recip,
}

pub struct DynExprBinary {
	pub kind: DynExprBinaryKind,
	pub lhs: Rc<DynExpr>,
	pub rhs: Rc<DynExpr>,
}

pub enum DynExprBinaryKind {
	Add,
	Sub,
	Mul,
}

pub enum DynExprKind {
	Arg(DynExprArg),
	Reduction(DynExprReduction),
	Unary(DynExprUnary),
	Binary(DynExprBinary),
}

impl DynExpr {
	pub fn find_reduction(&self) -> Option<&DynExprReduction> {
		#[rustfmt::skip]
		match &self.kind {
			DynExprKind::Arg(..) => None,
			DynExprKind::Reduction(r) => Some(r),
			DynExprKind::Unary(un) => un.expr.find_reduction(),

			// There should only be one reduction,
			// so it shouldn't happen that both sides return value.
			DynExprKind::Binary(bin) =>
				bin.lhs.find_reduction().or_else(|| bin.rhs.find_reduction())
			,
		}
	}

	pub fn post_reduce_common(&self) -> &Self {
		#[rustfmt::skip]
		match &self.kind {
			DynExprKind::Unary(un) => {
				if self.uses_elemwise_args {
					un.expr.post_reduce_common()
				} else {
					self
				}
			},
			DynExprKind::Binary(bin) => {
				if self.uses_elemwise_args {
					if bin.lhs.has_reduction {
						bin.lhs.post_reduce_common()
					} else {
						bin.rhs.post_reduce_common()
					}
				} else {
					self
				}
			},
			DynExprKind::Arg(..)
			| DynExprKind::Reduction(..) => {
				self
			},
		}
	}
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
	// - padding to the next multiple of HASH_WORD_SIZE with value 0
	// - expression key (EXPR_KEY_LEN bytes)
	// - padding to the next multiple of HASH_WORD_SIZE with value 0
	// Note that both DTypeId and ExprDiscriminant start from 1, so 0 is never a valid value.
	const DTYPE_CONFIG_BYTES: usize = Self::DTYPE_CONFIG_WORDS * HASH_WORD_SIZE;
	const EXPR_KEY_BYTES: usize = Self::EXPR_KEY_WORDS * HASH_WORD_SIZE;

	const DTYPE_CONFIG_WORDS: usize =
		(Self::DTYPE_CONFIG_LEN + 1).next_multiple_of(HASH_WORD_SIZE) / HASH_WORD_SIZE;
	const EXPR_KEY_WORDS: usize =
		Self::EXPR_KEY_LEN.next_multiple_of(HASH_WORD_SIZE) / HASH_WORD_SIZE;

	const KEY_WORDS: usize = Self::DTYPE_CONFIG_WORDS + Self::EXPR_KEY_WORDS;

	fn set_key(masks: InputMasks, reduce: bool, expr_id: &mut [HashWord], i: usize) -> usize;

	#[allow(clippy::indexing_slicing)]
	fn key() -> [HashWord; Self::KEY_WORDS] {
		let mut result = [HashWord::zero(); Self::KEY_WORDS];
		let expr_id = &mut result[Self::DTYPE_CONFIG_WORDS..];
		let len = Self::set_key(Self::MASKS, false, expr_id, 0);
		assert!(len == Self::EXPR_KEY_LEN);
		result
	}
}

pub trait ExprToDyn {
	fn to_dyn(e_mask: u64, r_mask: u64, s_mask: u64, reduce: bool) -> Rc<DynExpr>;
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

	fn set_key(masks: InputMasks, reduce: bool, expr_id: &mut [HashWord], i: usize) -> usize {
		let bit = 1_u64 << Idx;
		if reduce {
			let idx = (masks.reduce & (bit - 1)).count_ones() as usize;
			HashWord::set_byte(expr_id, i + 0, ExprDiscriminant::ReduceTensorArg as u8);
			HashWord::set_byte(expr_id, i + 1, idx as u8);
		} else {
			let idx = (masks.elemwise & (bit - 1)).count_ones() as usize;
			HashWord::set_byte(expr_id, i + 0, ExprDiscriminant::ElemwiseTensorArg as u8);
			HashWord::set_byte(expr_id, i + 1, idx as u8);
		}
		i + 2
	}
}

impl<const Idx: usize> ExprToDyn for TensorArg<Idx> {
	fn to_dyn(em: u64, rm: u64, _sm: u64, reduce: bool) -> Rc<DynExpr> {
		let bit = 1_u64 << Idx;
		if reduce {
			let index = (rm & (bit - 1)).count_ones() as usize;
			Rc::new(DynExpr {
				kind: DynExprKind::Arg(DynExprArg {
					kind: DynExprArgKind::ReduceTensor,
					index,
				}),
				is_reduction: false,
				has_reduction: false,
				uses_elemwise_args: false,
			})
		} else {
			let index = (em & (bit - 1)).count_ones() as usize;
			Rc::new(DynExpr {
				kind: DynExprKind::Arg(DynExprArg {
					kind: DynExprArgKind::ElemwiseTensor,
					index,
				}),
				is_reduction: false,
				has_reduction: false,
				uses_elemwise_args: true,
			})
		}
	}
}

impl<const Idx: usize> const ExprTrait for ScalarArg<Idx> {
	const MASKS: InputMasks = InputMasks { elemwise: 0, reduce: 0, scalar: 1 << Idx };

	const REDUCE_OP_COUNT: usize = 0;

	const EXPR_KEY_LEN: usize = 2;

	fn set_key(masks: InputMasks, _reduce: bool, expr_id: &mut [HashWord], i: usize) -> usize {
		let bit = 1_u64 << Idx;
		let idx = (masks.scalar & (bit - 1)).count_ones() as usize;
		HashWord::set_byte(expr_id, i + 0, ExprDiscriminant::ScalarArg as u8);
		HashWord::set_byte(expr_id, i + 1, idx as u8);
		i + 2
	}
}

impl<const Idx: usize> ExprToDyn for ScalarArg<Idx> {
	fn to_dyn(_e_mask: u64, _r_mask: u64, s_mask: u64, _reduce: bool) -> Rc<DynExpr> {
		let bit = 1_u64 << Idx;
		let index = (s_mask & (bit - 1)).count_ones() as usize;
		Rc::new(DynExpr {
			kind: DynExprKind::Arg(DynExprArg { kind: DynExprArgKind::Scalar, index }),
			is_reduction: false,
			has_reduction: false,
			uses_elemwise_args: false,
		})
	}
}

macro_rules! impl_expr_reduce {
	($name:ident, $red_name:ident) => {
		impl<A: const ExprTrait + ExprToDyn> const ExprTrait for $name<A> {
			const MASKS: InputMasks = InputMasks {
				elemwise: 0,
				reduce: A::MASKS.reduce | A::MASKS.elemwise,
				scalar: A::MASKS.scalar,
			};

			const REDUCE_OP_COUNT: usize = 1 + A::REDUCE_OP_COUNT;

			const EXPR_KEY_LEN: usize = 1 + A::EXPR_KEY_LEN;

			fn set_key(
				masks: InputMasks,
				_reduce: bool,
				expr_id: &mut [HashWord],
				i: usize,
			) -> usize {
				HashWord::set_byte(expr_id, i, ExprDiscriminant::$name as u8);
				A::set_key(masks, true, expr_id, i + 1)
			}
		}

		impl<A: const ExprTrait + ExprToDyn> ExprToDyn for $name<A> {
			fn to_dyn(e_mask: u64, r_mask: u64, s_mask: u64, reduce: bool) -> Rc<DynExpr> {
				assert!(!reduce);
				Rc::new(DynExpr {
					kind: DynExprKind::Reduction(DynExprReduction {
						kind: DynExprReductionKind::$red_name,
						expr: A::to_dyn(e_mask, r_mask, s_mask, true),
					}),
					is_reduction: true,
					has_reduction: true,
					uses_elemwise_args: false,
				})
			}
		}
	};
}

macro_rules! impl_expr_unary {
	($name:ident, $un_name:ident) => {
		impl<A: const ExprTrait + ExprToDyn> const ExprTrait for $name<A> {
			const MASKS: InputMasks = A::MASKS;

			const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT;

			const EXPR_KEY_LEN: usize = 1 + A::EXPR_KEY_LEN;

			fn set_key(
				masks: InputMasks,
				reduce: bool,
				expr_id: &mut [HashWord],
				i: usize,
			) -> usize {
				HashWord::set_byte(expr_id, i, ExprDiscriminant::$name as u8);
				A::set_key(masks, reduce, expr_id, i + 1)
			}
		}

		impl<A: const ExprTrait + ExprToDyn> ExprToDyn for $name<A> {
			fn to_dyn(e_mask: u64, r_mask: u64, s_mask: u64, reduce: bool) -> Rc<DynExpr> {
				let expr = A::to_dyn(e_mask, r_mask, s_mask, reduce);
				let e_has_reduction = expr.has_reduction;
				let e_uses_elemwise = expr.uses_elemwise_args;
				Rc::new(DynExpr {
					kind: DynExprKind::Unary(DynExprUnary {
						kind: DynExprUnaryKind::$un_name,
						expr,
					}),
					is_reduction: false,
					has_reduction: e_has_reduction,
					uses_elemwise_args: e_uses_elemwise,
				})
			}
		}
	};
}

macro_rules! impl_expr_binary {
	($name:ident, $bin_name:ident, $commutative:expr) => {
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

			fn set_key(
				masks: InputMasks,
				reduce: bool,
				expr_id: &mut [HashWord],
				i: usize,
			) -> usize {
				HashWord::set_byte(expr_id, i, ExprDiscriminant::$name as u8);
				let begin = i + 1;
				let mid = A::set_key(masks, reduce, expr_id, begin);
				assert!(mid == begin + A::EXPR_KEY_LEN);
				let end = B::set_key(masks, reduce, expr_id, mid);
				assert!(end == mid + B::EXPR_KEY_LEN);
				if !($commutative) {
					return end;
				}

				let mut swap = A::EXPR_KEY_LEN > B::EXPR_KEY_LEN;
				if A::EXPR_KEY_LEN == B::EXPR_KEY_LEN {
					let mut i = 0;
					while i < A::EXPR_KEY_LEN {
						let a = HashWord::get_byte(expr_id, begin + i);
						let b = HashWord::get_byte(expr_id, mid + i);
						if a != b {
							swap = a > b;
							break;
						}
						i += 1;
					}
				}

				if swap {
					let mid = B::set_key(masks, reduce, expr_id, begin);
					assert!(mid == begin + B::EXPR_KEY_LEN);
					let end = A::set_key(masks, reduce, expr_id, mid);
					assert!(end == mid + A::EXPR_KEY_LEN);
					end
				} else {
					end
				}
			}
		}

		impl<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn> ExprToDyn
			for $name<A, B>
		{
			fn to_dyn(e_mask: u64, r_mask: u64, s_mask: u64, reduce: bool) -> Rc<DynExpr> {
				let lhs = A::to_dyn(e_mask, r_mask, s_mask, reduce);
				let rhs = B::to_dyn(e_mask, r_mask, s_mask, reduce);
				let l_has_reduction = lhs.has_reduction;
				let r_has_reduction = rhs.has_reduction;
				let l_uses_elemwise = lhs.uses_elemwise_args;
				let r_uses_elemwise = rhs.uses_elemwise_args;
				Rc::new(DynExpr {
					kind: DynExprKind::Binary(DynExprBinary {
						kind: DynExprBinaryKind::$bin_name,
						lhs,
						rhs,
					}),
					is_reduction: false,
					has_reduction: l_has_reduction || r_has_reduction,
					uses_elemwise_args: l_uses_elemwise || r_uses_elemwise,
				})
			}
		}
	};
}

impl_expr_reduce!(SumExpr, Sum);
impl_expr_reduce!(MaxExpr, Max);

impl_expr_unary!(NegExpr, Neg);
impl_expr_unary!(ExpExpr, Exp);
impl_expr_unary!(LnExpr, Ln);
impl_expr_unary!(AbsExpr, Abs);
impl_expr_unary!(SqrtExpr, Sqrt);
impl_expr_unary!(RecipExpr, Recip);

impl_expr_binary!(AddExpr, Add, true);
impl_expr_binary!(SubExpr, Sub, false);
impl_expr_binary!(MulExpr, Mul, true);

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
	internal_dtype: DType,
	_expr: E,
}

pub struct DynKernelCall<'a> {
	pub key: &'a [HashWord],
	pub expr: &'a (dyn Fn() -> Rc<DynExpr> + 'a),
	pub output: &'a KernelOutput,
	pub tensor_args: &'a [KernelArg],
	pub reduce_count: usize,
	pub scalar_args: &'a [f64],
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
			internal_dtype,
			_expr: expr.0,
		}
	}

	pub fn call(&self, output: &Tensor) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); E::KEY_WORDS]:,
		[(); DynKernelCall::dtype_config_words(E::ELEMWISE_COUNT + E::REDUCE_COUNT)]:,
		[(); E::DTYPE_CONFIG_WORDS
			- DynKernelCall::dtype_config_words(E::ELEMWISE_COUNT + E::REDUCE_COUNT)]:,
		[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:,
	{
		let mut key = const { E::key() };
		let tensor_args: [&Tensor; E::ELEMWISE_COUNT + E::REDUCE_COUNT] =
			std::array::from_fn(|i| {
				if i < E::ELEMWISE_COUNT {
					self.elem_args[i]
				} else {
					self.reduce_args[i - E::ELEMWISE_COUNT]
				}
			});
		if E::REDUCE_COUNT == 0 {
			__run_elemwise_kernel::<{ E::ELEMWISE_COUNT }, { E::REDUCE_COUNT }>(
				&mut key,
				&|| E::to_dyn(E::ELEMWISE_MASK, E::REDUCE_MASK, E::SCALAR_MASK, false),
				output,
				tensor_args,
				&self.scalar_args,
				self.internal_dtype,
			)
		} else {
			__run_reduce_kernel::<{ E::ELEMWISE_COUNT }, { E::REDUCE_COUNT }>(
				&mut key,
				&|| E::to_dyn(E::ELEMWISE_MASK, E::REDUCE_MASK, E::SCALAR_MASK, false),
				output,
				tensor_args,
				&self.scalar_args,
				self.internal_dtype,
			)
		}
	}
}

impl<'a> DynKernelCall<'a> {
	#[inline]
	pub fn generate_expr(&self) -> Rc<DynExpr> {
		(self.expr)()
	}

	pub const fn dtype_config_items(tensor_count: usize) -> usize {
		// 0: Internal dtype
		// 1: Output dtype
		// 2..: `tensor_count` tensor dtypes
		(2 + tensor_count)
	}

	pub const fn dtype_config_words(tensor_count: usize) -> usize {
		(Self::dtype_config_items(tensor_count) * std::mem::size_of::<DTypeId>())
			.next_multiple_of(HASH_WORD_SIZE)
			/ HASH_WORD_SIZE
	}

	pub fn new_dtype_config<const T_CNT: usize>(
		internal_dtype: DType,
		output: &Tensor,
		tensors: [&Tensor; T_CNT],
	) -> [HashWord; Self::dtype_config_words(T_CNT)] {
		// TODO - we do indexing in set_byte()
		// check that there are no panics
		let mut result = [HashWord::zero(); Self::dtype_config_words(T_CNT)];
		HashWord::set_byte(&mut result, 0, internal_dtype.id().into());
		HashWord::set_byte(&mut result, 1, output.dtype().id().into());
		for (i, t) in tensors.iter().enumerate() {
			HashWord::set_byte(&mut result, 2 + i, t.dtype().id().into());
		}
		result
	}

	pub fn internal_dtype(&self) -> DType {
		assert!(self.key.len() >= Self::dtype_config_words(self.tensor_args.len()));
		// TODO: should use safe cast instead of `transmute()`
		let id: DTypeId = unsafe { std::mem::transmute(HashWord::get_byte(self.key, 0)) };
		id.to_dtype()
	}
	pub fn output_dtype(&self) -> DType {
		assert!(self.key.len() >= Self::dtype_config_words(self.tensor_args.len()));
		// TODO: should use safe cast instead of `transmute()`
		let id: DTypeId = unsafe { std::mem::transmute(HashWord::get_byte(self.key, 0)) };
		id.to_dtype()
	}
	pub fn arg_dtype(&self, i: usize) -> DType {
		assert!(i < self.tensor_args.len());
		assert!(self.key.len() >= Self::dtype_config_words(self.tensor_args.len()));
		let i = 2 + i;
		// TODO: should use safe cast instead of `transmute()`
		let id: DTypeId = unsafe { std::mem::transmute(HashWord::get_byte(self.key, i)) };
		id.to_dtype()
	}
}

#[inline(never)]
#[allow(clippy::indexing_slicing)]
fn __run_elemwise_kernel<'a, const E: usize, const R: usize>(
	key: &'a mut [HashWord],
	expr: &'a (dyn Fn() -> Rc<DynExpr> + 'a),
	output: &'a Tensor,
	tensor_args: [&'a Tensor; E + R],
	scalar_args: &'a [f64],
	internal_dtype: DType,
) -> Result<(), ErrPack<TensorOpError>>
where
	[(); 1 + E + R]:,
	[(); DynKernelCall::dtype_config_words(E + R)]:,
{
	let merged = DimMerger::<{ 1 + E + R }>::merge::<3>(std::array::from_fn(|i| {
		if i == 0 { output.map().dims() } else { tensor_args[i - 1].map().dims() }
	}))?;
	if merged.iter().any(|m| m.get(0).is_broadcasted()) {
		cold_path();
		return Err(CannotBroadcastOutputError.into());
	}

	let args: [KernelArg; E + R] = std::array::from_fn(|i| {
		let arg = tensor_args[i];
		let arg_dtype_bytes = arg.dtype().bytes();
		debug_assert!(arg_dtype_bytes > 0);
		KernelArg {
			stride_bytes: [
				merged[0].strides[1 + i] * arg_dtype_bytes,
				merged[1].strides[1 + i] * arg_dtype_bytes,
				merged[2].strides[1 + i] * arg_dtype_bytes,
			],
			offset_bytes: arg.map().offset() * arg_dtype_bytes,
			buf: arg.buf().device_ptr(),
		}
	});

	let out_dtype_bytes = output.dtype().bytes();
	debug_assert!(out_dtype_bytes > 0);
	let out = KernelOutput {
		size: [merged[0].size, merged[1].size, merged[2].size],
		stride_bytes: [
			merged[0].strides[0] * out_dtype_bytes,
			merged[1].strides[0] * out_dtype_bytes,
			merged[2].strides[0] * out_dtype_bytes,
		],
		offset_bytes: output.map().offset() * out_dtype_bytes,
		buf: output.buf().device_ptr(),
	};

	unsafe {
		let mut inp_fail = UnsafeBorrowFailFlag::new();
		let inp_borrows: [Option<BorrowGuard<DeviceBuffer>>; E + R] = std::array::from_fn(|i| {
			let tensor = tensor_args[i];
			let arg = &args[i];
			let same_as_output = std::ptr::eq(tensor.buf().as_ref(), output.buf().as_ref())
				&& likely(arg.offset_bytes == out.offset_bytes)
				&& likely(arg.stride_bytes[0] == out.stride_bytes[0] || out.size[0] <= 1)
				&& likely(arg.stride_bytes[1] == out.stride_bytes[1] || out.size[1] <= 1)
				&& likely(arg.stride_bytes[2] == out.stride_bytes[2] || out.size[2] <= 1);
			if same_as_output { None } else { Some(tensor.buf().unsafe_borrow(&mut inp_fail)) }
		});
		inp_fail.check()?;

		let out_borrow = output.buf().try_borrow_mut()?;

		// TODO - ensure_safe
		// TODO - ensure all on same device
		// TODO - other things may need to be checked before running the kernel

		let dtype_config =
			DynKernelCall::new_dtype_config::<{ E + R }>(internal_dtype, output, tensor_args);
		key[..dtype_config.len()].copy_from_slice(&dtype_config);

		output.device().run_elemwise_kernel(&DynKernelCall {
			key,
			expr,
			output: &out,
			tensor_args: &args,
			reduce_count: R,
			scalar_args,
		})?;

		std::mem::drop(out_borrow);
		std::mem::drop(inp_borrows);
	}
	Ok(())
}

#[inline(never)]
#[allow(clippy::indexing_slicing)]
#[allow(clippy::too_many_lines)]
fn __run_reduce_kernel<'a, const E: usize, const R: usize>(
	key: &'a mut [HashWord],
	expr: &'a (dyn Fn() -> Rc<DynExpr> + 'a),
	output: &'a Tensor,
	tensor_args: [&'a Tensor; E + R],
	scalar_args: &'a [f64],
	internal_dtype: DType,
) -> Result<(), ErrPack<TensorOpError>>
where
	[(); 1 + E + R]:,
	[(); DynKernelCall::dtype_config_words(E + R)]:,
{
	#[derive(Clone, Copy)]
	struct Split<'b> {
		top: SizeAndStride,
		batch: &'b [SizeAndStride],
	}
	fn split_last<'b>(tensor: &'b Tensor) -> Split<'b> {
		let dims = tensor.map().dims();
		if let Some((&top, batch)) = dims.split_last() {
			Split { top, batch }
		} else {
			cold_path();
			Split {
				top: SizeAndStride { size: 1, stride: 0 },
				batch: dims,
			}
		}
	}

	let output_split = split_last(output);
	let elem_args_split: [Split; E] = std::array::from_fn(|i| split_last(tensor_args[i]));
	let reduce_args_split: [Split; R] = std::array::from_fn(|i| split_last(tensor_args[E + i]));

	let reduce_args_top = DimMerger::<R>::merge_single_dim(reduce_args_split.map(|r| r.top))?;
	let post_reduce_top = DimMerger::<{ 1 + E }>::merge_single_dim(std::array::from_fn(|i| {
		if i == 0 { output_split.top } else { elem_args_split[i - 1].top }
	}))?;

	let batch_dims = DimMerger::<{ 1 + E + R }>::merge::<2>(std::array::from_fn(|i| {
		if i == 0 {
			output_split.batch
		} else if i <= E {
			elem_args_split[i - 1].batch
		} else {
			reduce_args_split[i - 1 - E].batch
		}
	}))?;

	let output_top = post_reduce_top.get(0);
	let output_batch = batch_dims.map(|m| m.get(0));
	if output_top.is_broadcasted() || output_batch.iter().any(SizeAndStride::is_broadcasted) {
		cold_path();
		return Err(CannotBroadcastOutputError.into());
	}
	let reduction_size = reduce_args_top.size;
	if output_top.size != 1 && output_top.size != reduction_size {
		cold_path();
		return Err(DimsDontMatchError.into());
	}

	let args: [KernelArg; E + R] = std::array::from_fn(|i| {
		let arg = tensor_args[i];
		let arg_dtype_bytes = arg.dtype().bytes();
		debug_assert!(arg_dtype_bytes > 0);
		let top_stride =
			if i < E { post_reduce_top.strides[1 + i] } else { reduce_args_top.strides[i - E] };
		KernelArg {
			stride_bytes: [
				batch_dims[0].strides[1 + i] * arg_dtype_bytes,
				batch_dims[1].strides[1 + i] * arg_dtype_bytes,
				top_stride * arg_dtype_bytes,
			],
			offset_bytes: arg.map().offset() * arg_dtype_bytes,
			buf: arg.buf().device_ptr(),
		}
	});

	let out_dtype_bytes = output.dtype().bytes();
	debug_assert!(out_dtype_bytes > 0);
	let out = KernelOutput {
		size: [batch_dims[0].size, batch_dims[1].size, reduction_size],
		stride_bytes: [
			batch_dims[0].strides[0] * out_dtype_bytes,
			batch_dims[1].strides[0] * out_dtype_bytes,
			if output_top.size == 1 { 0 } else { output_top.stride * out_dtype_bytes },
		],
		offset_bytes: output.map().offset() * out_dtype_bytes,
		buf: output.buf().device_ptr(),
	};

	unsafe {
		let mut inp_fail = UnsafeBorrowFailFlag::new();
		let inp_borrows: [Option<BorrowGuard<DeviceBuffer>>; E + R] = std::array::from_fn(|i| {
			let tensor = tensor_args[i];
			let arg = &args[i];
			let same_as_output = std::ptr::eq(tensor.buf().as_ref(), output.buf().as_ref())
				&& likely(arg.offset_bytes == out.offset_bytes)
				&& likely(arg.stride_bytes[0] == out.stride_bytes[0] || out.size[0] <= 1)
				&& likely(arg.stride_bytes[1] == out.stride_bytes[1] || out.size[1] <= 1)
				&& likely(arg.stride_bytes[2] == out.stride_bytes[2] || out.size[2] <= 1);
			if same_as_output { None } else { Some(tensor.buf().unsafe_borrow(&mut inp_fail)) }
		});
		inp_fail.check()?;

		let out_borrow = output.buf().try_borrow_mut()?;

		// TODO - ensure_safe
		// TODO - ensure all on same device
		// TODO - other things may need to be checked before running the kernel

		let dtype_config =
			DynKernelCall::new_dtype_config::<{ E + R }>(internal_dtype, output, tensor_args);
		key[..dtype_config.len()].copy_from_slice(&dtype_config);

		output.device().run_reduce_kernel(&DynKernelCall {
			key,
			expr,
			output: &out,
			tensor_args: &args,
			reduce_count: R,
			scalar_args,
		})?;

		std::mem::drop(out_borrow);
		std::mem::drop(inp_borrows);
	}
	Ok(())
}

impl<'a, E: const ExprTrait + ExprToDyn + Copy> EvaluatesToTensor for KernelCall<'a, E>
where
	[(); E::ELEMWISE_COUNT]:,
	[(); E::REDUCE_COUNT]:,
	[(); E::SCALAR_COUNT]:,
	[(); E::KEY_WORDS]:,
	[(); DynKernelCall::dtype_config_words(E::ELEMWISE_COUNT + E::REDUCE_COUNT)]:,
	[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:, // DTYPE_CONFIG_WORDS >= dtype_config_words()
	[(); E::DTYPE_CONFIG_WORDS
		- DynKernelCall::dtype_config_words(E::ELEMWISE_COUNT + E::REDUCE_COUNT)]:,
{
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.call(to)
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
