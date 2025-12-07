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

use crate::new::tensor::Tensor;
use crate::tensor::DType;

use compile::{DimConstraint, ShapeConstraint};

pub mod compile;
pub mod eval;

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct RcExpr {
	pub rc_expr: Rc<Expr>,
}

pub enum Expr {
	Input(ExprInput),
	Capture(ExprCapture),
	Cast(ExprCast),
	Unary(ExprUnary),
	Binary(ExprBinary),
	Reduction(ExprReduction),
	First(ExprFirst),
}

pub enum ExprInput {
	Tensor(Rc<ExprTensorRef>),
	Scalar(Rc<ExprScalarRef>),
}

// The tensor may be replaced before running the computation,
// but the dtype needs to be correct.
pub struct ExprTensorRef {
	pub tensor: RefCell<Option<Tensor>>,
	pub dtype: DType,
	pub shape_constraint: Vec<usize>,
	pub name: Option<Cow<'static, str>>,
}

pub struct ExprScalarRef {
	pub value: RefCell<Option<f64>>,
	pub name: Option<Cow<'static, str>>,
}

pub struct ExprCapture {
	pub expr: Rc<Expr>,
	pub tensor_ref: Rc<ExprTensorRef>,
}

pub struct ExprCast {
	pub expr: Rc<Expr>,
	pub dtype: DType,
}

pub struct ExprUnary {
	pub kind: ExprUnaryKind,
	pub expr: Rc<Expr>,
}

#[derive(Clone, Copy)]
pub enum ExprUnaryKind {
	Neg,
	Exp,
	Ln,
	Abs,
	Sqrt,
	Recip,
	Identity,
}

pub struct ExprFirst {
	pub lhs: Rc<Expr>,
	pub rhs: Rc<Expr>,
}

pub struct ExprBinary {
	pub kind: ExprBinaryKind,
	pub lhs: Rc<Expr>,
	pub rhs: Rc<Expr>,
}

#[derive(Clone, Copy)]
pub enum ExprBinaryKind {
	Add,
	Sub,
	Mul,
}

pub struct ExprReduction {
	pub kind: ExprReductionKind,
	pub expr: Rc<Expr>,
}

#[derive(Clone, Copy)]
pub enum ExprReductionKind {
	Sum,
	Max,
}

//--------------------------------------------------------------------------------------------------

impl ExprTensorRef {
	pub fn new(
		name: Option<Cow<'static, str>>,
		dtype: DType,
		shape: Vec<usize>,
	) -> Rc<ExprTensorRef> {
		Rc::new(ExprTensorRef {
			tensor: RefCell::new(None),
			dtype,
			shape_constraint: shape,
			name,
		})
	}

	pub fn shape_constraint(&self) -> ShapeConstraint {
		let source = if let Some(name) = &self.name { name.as_ref() } else { "unnamed tensor" };
		ShapeConstraint {
			constraint: self
				.shape_constraint
				.iter()
				.map(|&d| Some(DimConstraint { source: source.to_string(), size: d }))
				.collect(),
		}
	}
}

impl ExprScalarRef {
	pub fn new(name: Option<Cow<'static, str>>) -> Rc<ExprScalarRef> {
		Rc::new(ExprScalarRef { value: RefCell::new(None), name })
	}
}

//--------------------------------------------------------------------------------------------------

impl RcExpr {
	pub fn new_tensor_input(tensor_ref: Rc<ExprTensorRef>) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Input(ExprInput::Tensor(tensor_ref))),
		}
	}

	pub fn new_scalar_input(scalar_ref: Rc<ExprScalarRef>) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Input(ExprInput::Scalar(scalar_ref))),
		}
	}

	pub fn as_ref(&self) -> &Expr {
		&self.rc_expr
	}

	pub fn cast(self, dtype: DType) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Cast(ExprCast { expr: self.rc_expr, dtype })),
		}
	}

	pub fn exp(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Exp,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn ln(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Ln,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn abs(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Abs,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn sqrt(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Sqrt,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn recip(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Recip,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn sum(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Reduction(ExprReduction {
				kind: ExprReductionKind::Sum,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn max(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Reduction(ExprReduction {
				kind: ExprReductionKind::Max,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn first(first: RcExpr, second: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::First(ExprFirst { lhs: first.rc_expr, rhs: second.rc_expr })),
		}
	}

	pub fn capture(self, tensor_ref: Rc<ExprTensorRef>) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Capture(ExprCapture { expr: self.rc_expr, tensor_ref })),
		}
	}

	/*	pub fn compile(self) -> CompiledExpr {
		CompiledExpr::new(self)
	}*/
}

impl std::ops::Add for RcExpr {
	type Output = RcExpr;

	fn add(self, rhs: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Binary(ExprBinary {
				kind: ExprBinaryKind::Add,
				lhs: self.rc_expr,
				rhs: rhs.rc_expr,
			})),
		}
	}
}

impl std::ops::Sub for RcExpr {
	type Output = RcExpr;

	fn sub(self, rhs: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Binary(ExprBinary {
				kind: ExprBinaryKind::Sub,
				lhs: self.rc_expr,
				rhs: rhs.rc_expr,
			})),
		}
	}
}

impl std::ops::Mul for RcExpr {
	type Output = RcExpr;

	fn mul(self, rhs: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Binary(ExprBinary {
				kind: ExprBinaryKind::Mul,
				lhs: self.rc_expr,
				rhs: rhs.rc_expr,
			})),
		}
	}
}

impl std::ops::Neg for RcExpr {
	type Output = RcExpr;

	fn neg(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Neg,
				expr: self.rc_expr,
			})),
		}
	}
}

//--------------------------------------------------------------------------------------------------
