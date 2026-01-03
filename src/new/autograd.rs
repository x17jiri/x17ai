//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::new::expr::Expr;

//--------------------------------------------------------------------------------------------------

pub struct AutogradExpr {
	pub expr: Expr,
	pub backward_fn: Option<Box<dyn BackwardFn>>,
}

impl AutogradExpr {
	pub fn new(expr: Expr, backward_fn: Option<Box<dyn BackwardFn>>) -> Self {
		Self { expr, backward_fn }
	}

	pub fn unpack(self) -> (Expr, Option<Box<dyn BackwardFn>>) {
		(self.expr, self.backward_fn)
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
}

//--------------------------------------------------------------------------------------------------
