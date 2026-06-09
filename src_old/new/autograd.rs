//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::new::expr::{Expr, ToExpr};

//--------------------------------------------------------------------------------------------------

pub struct AutogradExpr {
	pub expr: Expr,
	pub backward_fn: Option<Box<dyn BackwardFn>>,
}

impl AutogradExpr {
	pub fn new<E: ToExpr>(expr: E, backward_fn: Option<Box<dyn BackwardFn>>) -> Self {
		Self { expr: expr.to_expr(), backward_fn }
	}

	pub fn unpack(self) -> (Expr, Option<Box<dyn BackwardFn>>) {
		(self.expr, self.backward_fn)
	}
}

impl From<Expr> for AutogradExpr {
	fn from(expr: Expr) -> Self {
		Self::new(expr, None)
	}
}

//--------------------------------------------------------------------------------------------------

pub trait BackwardFn {
	fn run(self: Box<Self>, d_out: Expr, autograd: &mut Autograd);
}

//--------------------------------------------------------------------------------------------------

pub struct Autograd {
	queue: Vec<(Box<dyn BackwardFn>, Expr)>,
	expr: Option<Expr>,
}

impl Default for Autograd {
	fn default() -> Self {
		Self::new()
	}
}

impl Autograd {
	pub fn new() -> Self {
		Self { queue: Vec::new(), expr: None }
	}

	pub fn enqueue(&mut self, node: Box<dyn BackwardFn>, d_out: Expr) {
		self.queue.push((node, d_out));
	}

	pub fn eval(&mut self, expr: Expr) {
		let prev_expr = self.expr.take();
		self.expr =
			Some(if let Some(prev_expr) = prev_expr { expr.first(prev_expr) } else { expr });
	}

	pub fn run<E: ToExpr>(backward_fn: Option<Box<dyn BackwardFn>>, grad: E) -> Expr {
		let grad = grad.to_expr();
		let Some(backward_fn) = backward_fn else {
			return grad;
		};
		let mut a = Self {
			queue: Vec::with_capacity(4),
			expr: Some(grad.clone()),
		};
		backward_fn.run(grad, &mut a);
		while let Some((node, d_out)) = a.queue.pop() {
			node.run(d_out, &mut a);
		}
		unsafe { a.expr.unwrap_unchecked() }
	}
}

//--------------------------------------------------------------------------------------------------
