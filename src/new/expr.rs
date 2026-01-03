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

pub enum ExprNode {
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

// The tensor may be replaced before running the computation,
// but the dtype needs to be correct.
pub struct TensorRef {
	pub name: Cow<'static, str>,
	pub dtype: DType,
	pub shape: Vec<usize>,
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
	pub dtype: DType,
}

pub struct ExprLabel {
	pub label: Cow<'static, str>,
	pub expr: Rc<ExprNode>,
}

pub struct ExprReshape {
	pub expr: Rc<ExprNode>,
	pub reshape_n: u8,
	pub reshape_to: ThinVec<usize>,
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
		shape: Vec<usize>,
		can_be_batched: CanBeBatched,
	) -> Rc<TensorRef> {
		Rc::new(TensorRef {
			dtype,
			shape,
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
			node: Rc::new(ExprNode::Input(ExprInput::Tensor(tensor_ref))),
		}
	}

	pub fn new_scalar_input(scalar_ref: Rc<ScalarRef>) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Input(ExprInput::Scalar(scalar_ref))),
		}
	}

	pub fn new_const(name: Cow<'static, str>, value: f64) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Const(ExprConst { name, value })),
		}
	}

	pub fn cast(self, dtype: DType) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Cast(ExprCast { expr: self.node, dtype })),
		}
	}

	pub fn label(self, label: Cow<'static, str>) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Label(ExprLabel { label, expr: self.node })),
		}
	}

	pub fn reshape(self, reshape_n: usize, reshape_to: &[usize]) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Reshape(ExprReshape {
				expr: self.node,
				reshape_n: u8::try_from(reshape_n).unwrap_or(255),
				reshape_to: ThinVec::from(reshape_to.to_vec()),
			})),
		}
	}

	pub fn exp(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::Exp,
				expr: self.node,
			})),
		}
	}

	pub fn ln(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary { kind: ExprUnaryKind::Ln, expr: self.node })),
		}
	}

	pub fn abs(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::Abs,
				expr: self.node,
			})),
		}
	}

	pub fn sqrt(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::Sqrt,
				expr: self.node,
			})),
		}
	}

	pub fn recip(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::Recip,
				expr: self.node,
			})),
		}
	}

	pub fn select_odd(self) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Unary(ExprUnary {
				kind: ExprUnaryKind::SelectOdd,
				expr: self.node,
			})),
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

	pub fn first(first: Expr, second: Expr) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Binary(ExprBinary {
				kind: ExprBinaryKind::First,
				lhs: first.node,
				rhs: second.node,
			})),
		}
	}

	pub fn capture(self, tensor_ref: Rc<TensorRef>) -> Expr {
		Expr {
			node: Rc::new(ExprNode::Capture(ExprCapture { expr: self.node, tensor_ref })),
		}
	}

	pub fn optional_capture(self, tensor_ref: Option<Rc<TensorRef>>) -> Expr {
		if let Some(tensor_ref) = tensor_ref { self.capture(tensor_ref) } else { self }
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
