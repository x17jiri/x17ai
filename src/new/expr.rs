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

use thin_vec::ThinVec;

use crate::tensor::DType;

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

	SumToMean,
}

pub struct ExprFirst {
	pub lhs: Rc<ExprNode>,
	pub rhs: Rc<ExprNode>,
}

pub struct ExprBinary {
	pub kind: ExprBinaryKind,
	pub lhs: Rc<ExprNode>,
	pub rhs: Rc<ExprNode>,
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
}

impl ScalarRef {
	pub fn new(name: Cow<'static, str>) -> Rc<ScalarRef> {
		Rc::new(ScalarRef { value: RefCell::new(None), name })
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

	pub fn cast(self, dtype: DType) -> Expr {
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

	pub fn select_odd(self) -> Expr {
		let mut local_errors = ThinVec::new();
		let mut shape = self.node.shape.clone();
		if let Some(last_dim) = Rc::make_mut(&mut shape).last() {
			if *last_dim % 2 != 0 {
				cold_path();
				local_errors.push(format!("select dimension not even"));
			}
			*last_dim /= 2;
		} else {
			cold_path();
			local_errors.push(format!("missing select dimension"));
			shape = Rc::from(&[0_usize])
		};
		Expr {
			node: Rc::new(ExprNode {
				dtype: self.node.dtype,
				shape,
				can_be_batched: self.node.can_be_batched,
				have_errors: self.node.have_errors,
				local_errors,
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::SelectOdd,
					expr: self.node,
				}),
			}),
		}
	}

	pub fn select_even(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::SelectEven,
				expr: self.node,
			})),
		}
	}

	pub fn sum(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::Sum,
				expr: self.node,
			})),
		}
	}

	pub fn sum_to_mean(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::SumToMean,
				expr: self.node,
			})),
		}
	}

	pub fn mean(self) -> Expr {
		self.clone().sum() * self.sum_to_mean()
	}

	pub fn max(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::Max,
				expr: self.node,
			})),
		}
	}

	pub fn first(self, second: Expr) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Binary(ExprBinary {
				kind: ExprBinaryKind::First,
				lhs: self.node,
				rhs: second.node,
			})),
		}
	}

	pub fn capture(self, tensor_ref: Rc<TensorRef>) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Capture(ExprCapture { expr: self.node, tensor_ref })),
		}
	}

	pub fn row_times_mat(self, mat: Expr) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Binary(ExprBinary {
				kind: ExprBinaryKind::RowTimesMat,
				lhs: self.node,
				rhs: mat.node,
			})),
		}
	}

	pub fn attention(self, kv: Expr) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Binary(ExprBinary {
				kind: ExprBinaryKind::Attention,
				lhs: self.node,
				rhs: kv.node,
			})),
		}
	}

	/*	pub fn compile(self) -> CompiledExpr {
		CompiledExpr::new(self)
	}*/
}

impl std::ops::Add for Expr {
	type Output = Expr;

	fn add(self, rhs: Expr) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Binary(ExprBinary {
				kind: ExprBinaryKind::Add,
				lhs: self.node,
				rhs: rhs.node,
			})),
		}
	}
}

impl std::ops::Sub for Expr {
	type Output = Expr;

	fn sub(self, rhs: Expr) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Binary(ExprBinary {
				kind: ExprBinaryKind::Sub,
				lhs: self.node,
				rhs: rhs.node,
			})),
		}
	}
}

impl std::ops::Mul for Expr {
	type Output = Expr;

	fn mul(self, rhs: Expr) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Binary(ExprBinary {
				kind: ExprBinaryKind::Mul,
				lhs: self.node,
				rhs: rhs.node,
			})),
		}
	}
}

impl std::ops::Neg for Expr {
	type Output = Expr;

	fn neg(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::Neg,
				expr: self.node,
			})),
		}
	}
}

//--------------------------------------------------------------------------------------------------
