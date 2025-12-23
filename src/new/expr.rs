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
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use thin_vec::ThinVec;

use crate::new::tensor::Tensor;
use crate::tensor::DType;

pub mod compile2;
pub mod eval;

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct RcExpr {
	pub rc_expr: Rc<Expr>,
}

pub struct Expr {
	kind: ExprKind,
	dtype: DType,
}

pub enum ExprKind {
	Input(ExprInput),
	Capture(ExprCapture),
	Cast(Rc<Expr>),
	Reshape(ExprReshape),
	SumToMean(Rc<Expr>),
	Select(ExprSelect),
	Unary(ExprUnary),
	Binary(ExprBinary),
	MatMul(ExprMatMul),
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
	pub value: Cell<Option<f64>>,
	pub dtype: DType,
	pub name: Option<Cow<'static, str>>,
}

pub struct ExprCapture {
	pub expr: Rc<Expr>,
	pub tensor_ref: Rc<ExprTensorRef>,
}

pub struct ExprReshape {
	pub expr: Rc<Expr>,
	pub reshape_n: u8,
	pub reshape_to: ThinVec<usize>,
}

pub struct ExprSelect {
	pub kind: ExprSelectKind,
	pub expr: Rc<Expr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExprSelectKind {
	Even,
	Odd,
}

pub struct ExprUnary {
	pub kind: ExprUnaryKind,
	pub expr: Rc<Expr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExprUnaryKind {
	Neg,
	Exp,
	Ln,
	Abs,
	Sqrt,
	Recip,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExprBinaryKind {
	Add,
	Sub,
	Mul,
}

impl ExprBinaryKind {
	pub fn is_commutative(&self) -> bool {
		match self {
			ExprBinaryKind::Add | ExprBinaryKind::Mul => true,
			ExprBinaryKind::Sub => false,
		}
	}
}

pub struct ExprMatMul {
	pub kind: ExprMatMulKind,
	pub lhs: Rc<Expr>,
	pub rhs: Rc<Expr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExprMatMulKind {
	RowTimesMat,
}

pub struct ExprReduction {
	pub kind: ExprReductionKind,
	pub expr: Rc<Expr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

impl ExprScalarRef {
	pub fn new(name: Option<Cow<'static, str>>, dtype: DType) -> Rc<ExprScalarRef> {
		Rc::new(ExprScalarRef { value: Cell::new(None), dtype, name })
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}
}

//--------------------------------------------------------------------------------------------------

impl RcExpr {
	pub fn new_tensor_input(tensor_ref: Rc<ExprTensorRef>) -> RcExpr {
		let dtype = tensor_ref.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Input(ExprInput::Tensor(tensor_ref)),
				dtype,
			}),
		}
	}

	pub fn new_scalar_input(scalar_ref: Rc<ExprScalarRef>) -> RcExpr {
		let dtype = scalar_ref.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Input(ExprInput::Scalar(scalar_ref)),
				dtype,
			}),
		}
	}

	pub fn dtype(&self) -> DType {
		self.rc_expr.dtype
	}

	pub fn as_ref(&self) -> &Expr {
		&self.rc_expr
	}

	pub fn cast(self, dtype: DType) -> RcExpr {
		let input_dtype = self.rc_expr.dtype;
		if input_dtype == dtype {
			self
		} else {
			RcExpr {
				rc_expr: Rc::new(Expr {
					kind: ExprKind::Cast(self.rc_expr),
					dtype,
				}),
			}
		}
	}

	pub fn reshape(self, reshape_n: usize, reshape_to: &[usize]) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Reshape(ExprReshape {
					expr: self.rc_expr,
					reshape_n: u8::try_from(reshape_n).unwrap_or(255),
					reshape_to: ThinVec::from(reshape_to.to_vec()),
				}),
				dtype,
			}),
		}
	}

	pub fn exp(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Exp,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}

	pub fn ln(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Ln,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}

	pub fn abs(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Abs,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}

	pub fn sqrt(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Sqrt,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}

	pub fn recip(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Recip,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}

	pub fn select_odd(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Select(ExprSelect {
					kind: ExprSelectKind::Odd,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}

	pub fn select_even(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Select(ExprSelect {
					kind: ExprSelectKind::Even,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}

	pub fn sum(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Reduction(ExprReduction {
					kind: ExprReductionKind::Sum,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}

	pub fn sum_to_mean(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::SumToMean(self.rc_expr),
				dtype,
			}),
		}
	}

	pub fn mean(self) -> RcExpr {
		self.clone().sum() * self.sum_to_mean()
	}

	pub fn max(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Reduction(ExprReduction {
					kind: ExprReductionKind::Max,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}

	pub fn first(first: RcExpr, second: RcExpr) -> RcExpr {
		let dtype = first.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::First(ExprFirst { lhs: first.rc_expr, rhs: second.rc_expr }),
				dtype,
			}),
		}
	}

	pub fn capture(self, tensor_ref: Rc<ExprTensorRef>) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Capture(ExprCapture { expr: self.rc_expr, tensor_ref }),
				dtype,
			}),
		}
	}

	pub fn row_times_mat(self, mat: RcExpr) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::MatMul(ExprMatMul {
					kind: ExprMatMulKind::RowTimesMat,
					lhs: self.rc_expr,
					rhs: mat.rc_expr,
				}),
				dtype,
			}),
		}
	}

	/*	pub fn compile(self) -> CompiledExpr {
		CompiledExpr::new(self)
	}*/
}

impl std::ops::Add for RcExpr {
	type Output = RcExpr;

	fn add(self, rhs: RcExpr) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::Add,
					lhs: self.rc_expr,
					rhs: rhs.rc_expr,
				}),
				dtype,
			}),
		}
	}
}

impl std::ops::Sub for RcExpr {
	type Output = RcExpr;

	fn sub(self, rhs: RcExpr) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::Sub,
					lhs: self.rc_expr,
					rhs: rhs.rc_expr,
				}),
				dtype,
			}),
		}
	}
}

impl std::ops::Mul for RcExpr {
	type Output = RcExpr;

	fn mul(self, rhs: RcExpr) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::Mul,
					lhs: self.rc_expr,
					rhs: rhs.rc_expr,
				}),
				dtype,
			}),
		}
	}
}

impl std::ops::Neg for RcExpr {
	type Output = RcExpr;

	fn neg(self) -> RcExpr {
		let dtype = self.rc_expr.dtype;
		RcExpr {
			rc_expr: Rc::new(Expr {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Neg,
					expr: self.rc_expr,
				}),
				dtype,
			}),
		}
	}
}

//--------------------------------------------------------------------------------------------------
