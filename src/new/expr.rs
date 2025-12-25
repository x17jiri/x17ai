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
pub struct Expr {
	pub node: Rc<ExprNode>,
}

pub struct ExprNode {
	kind: ExprKind,
	dtype: DType,
}

pub enum ExprKind {
	Input(ExprInput),
	Capture(ExprCapture),
	Reshape(ExprReshape),
	Unary(ExprUnary),
	Binary(ExprBinary),
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
	pub tensor: RefCell<Option<Tensor>>,
	pub dtype: DType,
	pub shape: Vec<usize>,
	pub batched: CanBeBatched,
	pub name: Option<Cow<'static, str>>,
}

pub struct ScalarRef {
	pub value: Cell<Option<f64>>,
	pub dtype: DType,
	pub name: Option<Cow<'static, str>>,
}

pub struct ExprCapture {
	pub expr: Rc<ExprNode>,
	pub tensor_ref: Rc<TensorRef>,
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
	Cast,

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
		name: Option<Cow<'static, str>>,
		dtype: DType,
		shape: Vec<usize>,
		batched: CanBeBatched,
	) -> Rc<TensorRef> {
		Rc::new(TensorRef {
			tensor: RefCell::new(None),
			dtype,
			shape,
			batched,
			name,
		})
	}
}

impl ScalarRef {
	pub fn new(name: Option<Cow<'static, str>>, dtype: DType) -> Rc<ScalarRef> {
		Rc::new(ScalarRef { value: Cell::new(None), dtype, name })
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}
}

//--------------------------------------------------------------------------------------------------

impl Expr {
	pub fn new_tensor_input(tensor_ref: Rc<TensorRef>) -> Expr {
		let dtype = tensor_ref.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Input(ExprInput::Tensor(tensor_ref)),
				dtype,
			}),
		}
	}

	pub fn new_scalar_input(scalar_ref: Rc<ScalarRef>) -> Expr {
		let dtype = scalar_ref.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Input(ExprInput::Scalar(scalar_ref)),
				dtype,
			}),
		}
	}

	pub fn dtype(&self) -> DType {
		self.node.dtype
	}

	pub fn as_ref(&self) -> &ExprNode {
		&self.node
	}

	pub fn cast(self, dtype: DType) -> Expr {
		let input_dtype = self.node.dtype;
		if input_dtype == dtype {
			self
		} else {
			Expr {
				node: Rc::new(ExprNode {
					kind: ExprKind::Unary(ExprUnary {
						kind: ExprUnaryKind::Cast,
						expr: self.node,
					}),
					dtype,
				}),
			}
		}
	}

	pub fn reshape(self, reshape_n: usize, reshape_to: &[usize]) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Reshape(ExprReshape {
					expr: self.node,
					reshape_n: u8::try_from(reshape_n).unwrap_or(255),
					reshape_to: ThinVec::from(reshape_to.to_vec()),
				}),
				dtype,
			}),
		}
	}

	pub fn exp(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Exp,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}

	pub fn ln(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary { kind: ExprUnaryKind::Ln, expr: self.node }),
				dtype,
			}),
		}
	}

	pub fn abs(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Abs,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}

	pub fn sqrt(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Sqrt,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}

	pub fn recip(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Recip,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}

	pub fn select_odd(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::SelectOdd,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}

	pub fn select_even(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::SelectEven,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}

	pub fn sum(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Sum,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}

	pub fn sum_to_mean(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::SumToMean,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}

	pub fn mean(self) -> Expr {
		self.clone().sum() * self.sum_to_mean()
	}

	pub fn max(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Max,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}

	pub fn first(first: Expr, second: Expr) -> Expr {
		let dtype = first.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::First,
					lhs: first.node,
					rhs: second.node,
				}),
				dtype,
			}),
		}
	}

	pub fn capture(self, tensor_ref: Rc<TensorRef>) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Capture(ExprCapture { expr: self.node, tensor_ref }),
				dtype,
			}),
		}
	}

	pub fn row_times_mat(self, mat: Expr) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::RowTimesMat,
					lhs: self.node,
					rhs: mat.node,
				}),
				dtype,
			}),
		}
	}

	pub fn attention(self, kv: Expr) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::Attention,
					lhs: self.node,
					rhs: kv.node,
				}),
				dtype,
			}),
		}
	}

	/*	pub fn compile(self) -> CompiledExpr {
		CompiledExpr::new(self)
	}*/
}

impl std::ops::Add for Expr {
	type Output = Expr;

	fn add(self, rhs: Expr) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::Add,
					lhs: self.node,
					rhs: rhs.node,
				}),
				dtype,
			}),
		}
	}
}

impl std::ops::Sub for Expr {
	type Output = Expr;

	fn sub(self, rhs: Expr) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::Sub,
					lhs: self.node,
					rhs: rhs.node,
				}),
				dtype,
			}),
		}
	}
}

impl std::ops::Mul for Expr {
	type Output = Expr;

	fn mul(self, rhs: Expr) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Binary(ExprBinary {
					kind: ExprBinaryKind::Mul,
					lhs: self.node,
					rhs: rhs.node,
				}),
				dtype,
			}),
		}
	}
}

impl std::ops::Neg for Expr {
	type Output = Expr;

	fn neg(self) -> Expr {
		let dtype = self.node.dtype;
		Expr {
			node: Rc::new(ExprNode {
				kind: ExprKind::Unary(ExprUnary {
					kind: ExprUnaryKind::Neg,
					expr: self.node,
				}),
				dtype,
			}),
		}
	}
}

//--------------------------------------------------------------------------------------------------
