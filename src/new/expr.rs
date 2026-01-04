//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#![allow(clippy::use_self)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::implicit_hasher)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::new_without_default)]
#![allow(clippy::cast_possible_wrap)]

use std::borrow::Cow;
use std::cell::RefCell;
use std::hint::cold_path;
use std::rc::Rc;

use safetensors::tensor;
use thin_vec::ThinVec;

use crate::tensor::DType;
use crate::tensor::device::dtype::common_dtype;
use crate::util::LossyFrom;

pub mod compile2;
pub mod eval;

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct Expr {
	pub node: Rc<ExprNode>,
}

pub struct ExprNode {
	kind: ExprKind,
	dtype: Option<DType>,
	shape: Rc<[usize]>,
	can_be_batched: bool,
	have_errors: bool,
	local_errors: ThinVec<String>,
}

impl ExprNode {
	pub fn dtype(&self) -> Option<DType> {
		self.dtype
	}

	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	pub fn can_be_batched(&self) -> bool {
		self.can_be_batched
	}

	pub fn have_errors(&self) -> bool {
		self.have_errors
	}
}

pub enum ExprKind {
	Const(ExprConst),
	Input(ExprInput),
	Capture(ExprCapture),
	Cast(ExprCast),
	Label(ExprLabel),
	Reshape(ExprReshape),
	Unary(ExprUnary),
	Binary(ExprBinary),
}

pub struct ExprConst {
	pub name: Cow<'static, str>,
	pub value: f64,
}
pub enum ExprInput {
	Tensor(Rc<TensorRef>),
	Scalar(Rc<ScalarRef>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanBeBatched {
	No,
	Yes,
}

pub struct TensorRef {
	pub name: Cow<'static, str>,
	pub dtype: DType,
	pub shape: Rc<[usize]>,
	pub can_be_batched: bool,
}

pub struct ScalarRef {
	pub value: RefCell<Option<f64>>,
	pub name: Cow<'static, str>,
}

pub struct ExprCapture {
	pub expr: Rc<ExprNode>,
	pub tensor_ref: Rc<TensorRef>,
}

pub struct ExprCast {
	pub expr: Rc<ExprNode>,
}

pub struct ExprLabel {
	pub label: Cow<'static, str>,
	pub expr: Rc<ExprNode>,
}

pub struct ExprReshape {
	pub expr: Rc<ExprNode>,
}

pub struct ExprUnary {
	pub kind: ExprUnaryKind,
	pub expr: Rc<ExprNode>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExprUnaryKind {
	NoOp,
	Neg,
	Exp,
	Ln,
	Abs,
	Sqrt,
	Recip,

	Sum,
	Max,

	SelectEven,
	SelectOdd,
}

pub struct ExprFirst {
	pub lhs: Rc<ExprNode>,
	pub rhs: Rc<ExprNode>,
}

pub struct ExprBinary {
	pub kind: ExprBinaryKind,
	pub lhs: Rc<ExprNode>,
	pub rhs: Rc<ExprNode>,
	pub lhs_broadcasted: bool,
	pub rhs_broadcasted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExprBinaryKind {
	Add,
	Sub,
	Mul,

	First,
	RowTimesMat,
	Attention,
}

//--------------------------------------------------------------------------------------------------

pub trait ToExpr {
	fn to_expr(self) -> Expr;
}

impl ToExpr for Expr {
	fn to_expr(self) -> Expr {
		self
	}
}

//--------------------------------------------------------------------------------------------------

impl TensorRef {
	pub fn new(
		name: Cow<'static, str>,
		dtype: DType,
		shape: &[usize],
		can_be_batched: CanBeBatched,
	) -> Rc<TensorRef> {
		Rc::new(TensorRef {
			dtype,
			shape: Rc::from(shape),
			can_be_batched: can_be_batched != CanBeBatched::No,
			name,
		})
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	pub fn can_be_batched(&self) -> bool {
		self.can_be_batched
	}
}

impl ToExpr for Rc<TensorRef> {
	fn to_expr(self) -> Expr {
		Expr::new_tensor_input(self)
	}
}

impl ScalarRef {
	pub fn new(name: Cow<'static, str>) -> Rc<ScalarRef> {
		Rc::new(ScalarRef { value: RefCell::new(None), name })
	}
}

impl ToExpr for Rc<ScalarRef> {
	fn to_expr(self) -> Expr {
		Expr::new_scalar_input(self)
	}
}

//--------------------------------------------------------------------------------------------------

impl Expr {
	pub fn new_tensor_input(tensor_ref: Rc<TensorRef>) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: Some(tensor_ref.dtype),
				shape: tensor_ref.shape.clone(),
				can_be_batched: tensor_ref.can_be_batched,
				have_errors: false,
				local_errors: ThinVec::new(),
				kind: ExprKind::Input(ExprInput::Tensor(tensor_ref)),
			}),
		}
	}

	pub fn new_scalar_input(scalar_ref: Rc<ScalarRef>) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: None,
				shape: Rc::from([]),
				can_be_batched: false,
				have_errors: false,
				local_errors: ThinVec::new(),
				kind: ExprKind::Input(ExprInput::Scalar(scalar_ref)),
			}),
		}
	}

	pub fn new_const(name: Cow<'static, str>, value: f64) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: None,
				shape: Rc::from([]),
				can_be_batched: false,
				have_errors: false,
				local_errors: ThinVec::new(),
				kind: ExprKind::Const(ExprConst { name, value }),
			}),
		}
	}

	pub fn dtype(&self) -> Option<DType> {
		self.node.dtype
	}

	pub fn shape(&self) -> &[usize] {
		&self.node.shape
	}

	pub fn can_be_batched(&self) -> bool {
		self.node.can_be_batched
	}

	pub fn have_errors(&self) -> bool {
		self.node.have_errors
	}

	pub fn cast(self, dtype: DType) -> Expr {
		if self.node.dtype == Some(dtype) {
			return self;
		}
		Expr {
			node: Rc::new(ExprNode {
				dtype: Some(dtype),
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors: ThinVec::new(),
				kind: ExprKind::Cast(ExprCast { expr: self.node }),
			}),
		}
	}

	pub fn label(self, label: Cow<'static, str>) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors: ThinVec::new(),
				kind: ExprKind::Label(ExprLabel { label, expr: self.node }),
			}),
		}
	}

	pub fn reshape_n(self, n_replace: usize, replace_with: &[usize]) -> Expr {
		let old_shape = self.node.shape();
		let n_keep = old_shape.len().saturating_sub(n_replace);
		let (keep, replace) = old_shape.split_at(n_keep);
		if replace == replace_with {
			return self;
		}

		let old_elems = replace.iter().product::<usize>();
		let new_elems = replace_with.iter().product::<usize>();

		let mut local_errors = ThinVec::new();
		if old_elems != new_elems {
			local_errors.push(format!(
				"Reshape: element count mismatch (got {new_elems}, expected {old_elems})",
			));
		}

		let mut new_shape = Vec::with_capacity(keep.len() + replace_with.len());
		new_shape.extend_from_slice(keep);
		new_shape.extend_from_slice(replace_with);
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: Rc::from(&new_shape[..]),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors || !local_errors.is_empty(),
				local_errors,
				kind: ExprKind::Reshape(ExprReshape { expr: self.node }),
			}),
		}
	}

	pub fn reshape(self, new_shape: &[usize]) -> Expr {
		let old_shape = self.node.shape();
		if old_shape == new_shape {
			return self;
		}

		let old_elems = old_shape.iter().product::<usize>();
		let new_elems = new_shape.iter().product::<usize>();

		let mut local_errors = ThinVec::new();
		if old_elems != new_elems {
			local_errors.push(format!(
				"Reshape: element count mismatch (got {new_elems}, expected {old_elems})",
			));
		}

		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: Rc::from(new_shape),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors || !local_errors.is_empty(),
				local_errors,
				kind: ExprKind::Reshape(ExprReshape { expr: self.node }),
			}),
		}
	}

	pub fn exp(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors: ThinVec::new(),
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Exp,
					expr: self.node,
				}),
			}),
		}
	}

	pub fn ln(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors: ThinVec::new(),
				kind: ExprKind::Unary(ExprUnary { kind: ExprUnaryKind::Ln, expr: self.node }),
			}),
		}
	}

	pub fn abs(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors: ThinVec::new(),
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Abs,
					expr: self.node,
				}),
			}),
		}
	}

	pub fn sqrt(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors: ThinVec::new(),
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Sqrt,
					expr: self.node,
				}),
			}),
		}
	}

	pub fn recip(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors: ThinVec::new(),
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Recip,
					expr: self.node,
				}),
			}),
		}
	}

	fn __select(self, kind: ExprUnaryKind) -> Expr {
		let mut local_errors = ThinVec::new();
		let mut shape = self.node.shape.clone();
		if let Some(last_dim) = Rc::make_mut(&mut shape).last_mut() {
			if *last_dim % 2 != 0 {
				cold_path();
				local_errors.push(format!("select dimension not even"));
			}
			*last_dim /= 2;
		} else {
			cold_path();
			local_errors.push(format!("missing select dimension"));
			let shape_slice: &[usize] = &[0];
			shape = Rc::from(shape_slice);
		}
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape,
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors || !local_errors.is_empty(),
				local_errors,
				kind: ExprKind::Unary(ExprUnary { kind, expr: self.node }),
			}),
		}
	}

	pub fn select_odd(self) -> Expr {
		self.__select(ExprUnaryKind::SelectOdd)
	}

	pub fn select_even(self) -> Expr {
		self.__select(ExprUnaryKind::SelectEven)
	}

	fn __reduce(self, kind: ExprUnaryKind) -> Expr {
		let mut local_errors = ThinVec::new();
		let mut shape = self.node.shape.clone();
		if let Some(last_dim) = Rc::make_mut(&mut shape).last_mut() {
			*last_dim = 1;
		} else {
			cold_path();
			local_errors.push(format!("missing reduce dimension"));
			let shape_slice: &[usize] = &[1];
			shape = Rc::from(shape_slice);
		}
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape,
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors || !local_errors.is_empty(),
				local_errors,
				kind: ExprKind::Unary(ExprUnary { kind, expr: self.node }),
			}),
		}
	}

	pub fn max(self) -> Expr {
		self.__reduce(ExprUnaryKind::Max)
	}

	pub fn sum(self) -> Expr {
		self.__reduce(ExprUnaryKind::Sum)
	}

	pub fn sum_to_mean(&self) -> Expr {
		let shape = self.node.shape();
		let last_dim = shape.last().copied().unwrap_or(1);
		let c = 1.0 / f64::lossy_from(last_dim);
		Expr::new_const(format!("1.0 / {last_dim}").into(), c)
	}

	pub fn mean(self) -> Expr {
		self.clone().sum() * self.sum_to_mean()
	}

	pub fn first(self, second: Expr) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors || second.node.have_errors,
				local_errors: ThinVec::new(),
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::First,
					lhs: self.node,
					rhs: second.node,
					lhs_broadcasted: false,
					rhs_broadcasted: false,
				}),
			}),
		}
	}

	pub fn capture(self, tensor_ref: Rc<TensorRef>) -> Expr {
		let mut local_errors = ThinVec::new();
		if let Some(node_dtype) = self.node.dtype
			&& node_dtype != tensor_ref.dtype
		{
			cold_path();
			local_errors.push(format!(
				"capture: dtype mismatch (got {}, expected {})",
				node_dtype, tensor_ref.dtype
			));
		}
		if self.node.shape() != tensor_ref.shape()
			|| self.node.can_be_batched() != tensor_ref.can_be_batched()
		{
			cold_path();
			local_errors.push(format!(
				"capture: shape mismatch (got {}, expected {})",
				shape_to_str(self.node.can_be_batched(), self.node.shape()),
				shape_to_str(tensor_ref.can_be_batched(), tensor_ref.shape()),
			));
		}
		Expr {
			node: Rc::new(ExprNode {
				dtype: Some(tensor_ref.dtype),
				shape: tensor_ref.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors,
				kind: ExprKind::Capture(ExprCapture { expr: self.node, tensor_ref }),
			}),
		}
	}

	pub fn row_times_mat(self, mat: Expr) -> Expr {
		let mut local_errors = ThinVec::new();

		let r_shape = self.node.shape();
		let m_shape = mat.node.shape();
		if r_shape.len() < 1 || m_shape.len() < 2 {
			cold_path();
			local_errors.push(format!("matmul: not enough dimensions"));
		}
		let (r_rest, [r_len]) = split_shape::<1>(r_shape);
		let (m_rest, [m_row, m_col]) = split_shape::<2>(m_shape);
		if r_len != m_row {
			cold_path();
			local_errors.push(format!("matmul: shape mismatch"));
		}
		let mut shape = Rc::<[usize]>::from(r_shape);
		*Rc::make_mut(&mut shape).last_mut().unwrap() = m_col;

		if !m_rest.is_empty() || mat.node.can_be_batched {
			cold_path();
			local_errors.push("row times mat: mat cannot be batched".into());
		}
		let m_broadcasted = !r_rest.is_empty();

		let dtype = same_dtype(self.node.dtype(), mat.node.dtype(), &mut local_errors);

		Expr {
			node: Rc::new(ExprNode {
				dtype,
				shape: Rc::from(&shape[..]),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors
					|| mat.node.have_errors
					|| !local_errors.is_empty(),
				local_errors,
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::RowTimesMat,
					lhs: self.node,
					rhs: mat.node,
					lhs_broadcasted: false,
					rhs_broadcasted: m_broadcasted,
				}),
			}),
		}
	}

	pub fn attention(self, kv: Expr) -> Expr {
		let mut local_errors = ThinVec::new();
		let q_shape = self.node.shape();
		let kv_shape = kv.node.shape();
		if q_shape.len() < 3 || kv_shape.len() < 3 {
			cold_path();
			local_errors.push(format!("attention: not enough dimensions"));
		}
		let (q_rest, [q1, q2, q3]) = split_shape::<3>(q_shape);
		let (kv_rest, [kv1, kv2, kv3]) = split_shape::<3>(kv_shape);
		if kv1 != 1 || q2 != kv2 || q3 >= kv3 {
			cold_path();
			local_errors.push(format!("attention: shape mismatch"));
		}
		let (mut shape, is_broadcasted) = broadcast_shapes(q_rest, kv_rest, &mut local_errors);
		shape.push(q1);
		shape.push(q2);
		shape.push(kv3.saturating_sub(q3));
		if is_broadcasted[0] || is_broadcasted[1] {
			cold_path();
			local_errors.push(format!("attention inputs cannot be broadcasted"));
		}

		let dtype = same_dtype(self.node.dtype(), kv.node.dtype(), &mut local_errors);

		Expr {
			node: Rc::new(ExprNode {
				dtype,
				shape: Rc::from(&shape[..]),
				can_be_batched: self.node.can_be_batched || kv.node.can_be_batched,
				have_errors: self.node.have_errors
					|| kv.node.have_errors
					|| !local_errors.is_empty(),
				local_errors,
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::Attention,
					lhs: self.node,
					rhs: kv.node,
					lhs_broadcasted: false,
					rhs_broadcasted: false,
				}),
			}),
		}
	}

	fn __binary_op(self, kind: ExprBinaryKind, rhs: Expr) -> Expr {
		let mut local_errors = ThinVec::new();
		let dtype = same_dtype(self.node.dtype, rhs.node.dtype, &mut local_errors);
		let (shape, is_broadcasted) =
			broadcast_shapes(self.node.shape(), rhs.node.shape(), &mut local_errors);
		Expr {
			node: Rc::new(ExprNode {
				dtype,
				shape: Rc::from(&shape[..]),
				can_be_batched: self.node.can_be_batched || rhs.node.can_be_batched,
				have_errors: self.node.have_errors
					|| rhs.node.have_errors
					|| !local_errors.is_empty(),
				local_errors,
				kind: ExprKind::Binary(ExprBinary {
					kind,
					lhs: self.node,
					rhs: rhs.node,
					lhs_broadcasted: is_broadcasted[0],
					rhs_broadcasted: is_broadcasted[1],
				}),
			}),
		}
	}

	pub fn log_error(self, msg: String) -> Expr {
		let mut local_errors = ThinVec::new();
		local_errors.push(msg);
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: true,
				local_errors,
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::NoOp,
					expr: self.node,
				}),
			}),
		}
	}
}

impl std::ops::Add for Expr {
	type Output = Expr;

	fn add(self, rhs: Expr) -> Expr {
		self.__binary_op(ExprBinaryKind::Add, rhs)
	}
}

impl std::ops::Sub for Expr {
	type Output = Expr;

	fn sub(self, rhs: Expr) -> Expr {
		self.__binary_op(ExprBinaryKind::Sub, rhs)
	}
}

impl std::ops::Mul for Expr {
	type Output = Expr;

	fn mul(self, rhs: Expr) -> Expr {
		self.__binary_op(ExprBinaryKind::Mul, rhs)
	}
}

impl std::ops::Neg for Expr {
	type Output = Expr;

	fn neg(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape: self.node.shape.clone(),
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors: ThinVec::new(),
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Neg,
					expr: self.node,
				}),
			}),
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub fn split_shape<const N: usize>(shape: &[usize]) -> (&[usize], [usize; N]) {
	let len = shape.len();
	let cnt = len.min(N);
	let rest = len - cnt;
	let mut a = [1; N];
	for i in 0..N {
		if i < cnt {
			a[N - 1 - i] = shape[len - 1 - i];
		}
	}
	(&shape[..rest], a)
}

//--------------------------------------------------------------------------------------------------

fn same_dtype(
	a_dtype: Option<DType>,
	b_dtype: Option<DType>,
	err: &mut ThinVec<String>,
) -> Option<DType> {
	match (a_dtype, b_dtype) {
		(None, None) => None,
		(Some(a_dt), None) => Some(a_dt),
		(None, Some(b_dt)) => Some(b_dt),
		(Some(a_dt), Some(b_dt)) => {
			if a_dt == b_dt {
				Some(a_dt)
			} else {
				cold_path();
				err.push(format!("dtype mismatch: {} vs {}", a_dt, b_dt));
				Some(common_dtype(a_dt, b_dt))
			}
		},
	}
}

//--------------------------------------------------------------------------------------------------

pub fn broadcast_shapes(
	a: &[usize],
	b: &[usize],
	err: &mut ThinVec<String>,
) -> (Vec<usize>, [bool; 2]) {
	let mut is_broadcasted = [false, false];
	let mut result = Vec::new();
	let len = a.len().max(b.len());
	let skip_a = len - a.len();
	let skip_b = len - b.len();
	for d in 0..len {
		let dim_a = if d < skip_a { 1 } else { a[d - skip_a] };
		let dim_b = if d < skip_b { 1 } else { b[d - skip_b] };
		let dim = if dim_a == dim_b {
			dim_a
		} else if dim_b == 1 {
			is_broadcasted[1] = true;
			dim_a
		} else if dim_a == 1 {
			is_broadcasted[0] = true;
			dim_b
		} else {
			cold_path();
			err.push(format!("broadcast dimension mismatch: {:?} vs {:?}", dim_a, dim_b));
			dim_a.max(dim_b)
		};
		result.push(dim);
	}
	(result, is_broadcasted)
}

//--------------------------------------------------------------------------------------------------

pub fn shape_to_str(can_be_batched: bool, shape: &[usize]) -> String {
	let mut result = String::from("[");
	if can_be_batched {
		result.push_str("*, ");
	}
	for (i, &dim) in shape.iter().enumerate() {
		if i > 0 {
			result.push_str(", ");
		}
		result.push_str(&dim.to_string());
	}
	result.push(']');
	result
}

//--------------------------------------------------------------------------------------------------
