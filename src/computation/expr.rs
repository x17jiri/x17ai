//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#![allow(clippy::use_self)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::tensor::{DType, Tensor};

//--------------------------------------------------------------------------------------------------

pub struct Expr {
	pub kind: Rc<ExprKind>,
}

pub enum ExprKind {
	Input(ExprInput),
	Output(ExprOutput),
	Cast(ExprCast),
	Unary(ExprUnary),
	Binary(ExprBinary),
	Reduction(ExprReduction),
}

pub enum ExprInput {
	Tensor(Rc<ExprInputTensor>),
	Scalar(Rc<ExprInputScalar>),
}

pub struct ExprInputTensor {
	pub tensor: RefCell<Option<Tensor>>,
	pub dtype: DType,
}

pub struct ExprInputScalar {
	pub value: RefCell<Option<f64>>,
	pub dtype: DType,
}

pub struct ExprOutput {
	pub expr: Expr,
	pub dtype: DType,
	pub value: RefCell<Option<Tensor>>,
}

pub struct ExprCast {
	pub expr: Expr,
	pub dtype: DType,
}

pub struct ExprUnary {
	pub kind: ExprUnaryKind,
	pub expr: Expr,
}

pub enum ExprUnaryKind {
	Neg,
	Exp,
	Ln,
	Abs,
	Sqrt,
	Recip,
}

pub struct ExprBinary {
	pub kind: ExprBinaryKind,
	pub lhs: Expr,
	pub rhs: Expr,
}

pub enum ExprBinaryKind {
	Add,
	Sub,
	Mul,
}

pub struct ExprReduction {
	pub kind: ExprReductionKind,
	pub expr: Expr,
}

pub enum ExprReductionKind {
	Sum,
	Max,
}

//--------------------------------------------------------------------------------------------------

impl Expr {
	pub fn new_tensor_input(dtype: DType) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Input(ExprInput::Tensor(Rc::new(ExprInputTensor {
				tensor: RefCell::new(None),
				dtype,
			})))),
		}
	}

	pub fn new_scalar_input(dtype: DType) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Input(ExprInput::Scalar(Rc::new(ExprInputScalar {
				value: RefCell::new(None),
				dtype,
			})))),
		}
	}

	pub fn cast(self, dtype: DType) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Cast(ExprCast { expr: self, dtype })),
		}
	}

	pub fn exp(self) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Unary(ExprUnary { kind: ExprUnaryKind::Exp, expr: self })),
		}
	}

	pub fn ln(self) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Unary(ExprUnary { kind: ExprUnaryKind::Ln, expr: self })),
		}
	}

	pub fn abs(self) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Unary(ExprUnary { kind: ExprUnaryKind::Abs, expr: self })),
		}
	}

	pub fn sqrt(self) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Unary(ExprUnary { kind: ExprUnaryKind::Sqrt, expr: self })),
		}
	}

	pub fn recip(self) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Unary(ExprUnary { kind: ExprUnaryKind::Recip, expr: self })),
		}
	}

	pub fn sum(self) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Reduction(ExprReduction {
				kind: ExprReductionKind::Sum,
				expr: self,
			})),
		}
	}

	pub fn max(self) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Reduction(ExprReduction {
				kind: ExprReductionKind::Max,
				expr: self,
			})),
		}
	}
}

impl std::ops::Add for Expr {
	type Output = Expr;

	fn add(self, rhs: Expr) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Binary(ExprBinary {
				kind: ExprBinaryKind::Add,
				lhs: self,
				rhs,
			})),
		}
	}
}

impl std::ops::Sub for Expr {
	type Output = Expr;

	fn sub(self, rhs: Expr) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Binary(ExprBinary {
				kind: ExprBinaryKind::Sub,
				lhs: self,
				rhs,
			})),
		}
	}
}

impl std::ops::Mul for Expr {
	type Output = Expr;

	fn mul(self, rhs: Expr) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Binary(ExprBinary {
				kind: ExprBinaryKind::Mul,
				lhs: self,
				rhs,
			})),
		}
	}
}

impl std::ops::Neg for Expr {
	type Output = Expr;

	fn neg(self) -> Expr {
		Expr {
			kind: Rc::new(ExprKind::Unary(ExprUnary { kind: ExprUnaryKind::Neg, expr: self })),
		}
	}
}

//--------------------------------------------------------------------------------------------------
