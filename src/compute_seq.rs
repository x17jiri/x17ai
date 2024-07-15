// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::dtype::DType;
use crate::expr::{Expr, ExprKind};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

struct _PostorderItem {
	expr: *const Expr,
	cache_key: String,
	inputs: Vec<usize>,
	ref_count: usize,
}

pub struct PostorderItem<'e> {
	pub expr: &'e Expr,
	pub cache_key: &'e str,
	pub inputs: &'e [usize],
	pub ref_count: usize,
}

pub struct ComputeSequence {
	pub expr: Rc<Expr>,

	// SAFETY: We have a shared ownership of `expr` and since it is immutable,
	// as long as we don't drop it, all pointers to sub-expressions are valid.
	roots: HashMap<*const Expr, isize>,
	postorder: Vec<_PostorderItem>,
	processed: HashMap<*const Expr, bool>,

	swapped_operands: HashSet<*const Expr>,
}

pub struct ComputeSequenceIter<'a> {
	postorder_iter: std::slice::Iter<'a, _PostorderItem>,
}

impl<'e> Iterator for ComputeSequenceIter<'e> {
	type Item = PostorderItem<'e>;

	fn next(&mut self) -> Option<Self::Item> {
		self.postorder_iter.next().map(|item| PostorderItem {
			expr: unsafe { &*item.expr },
			cache_key: &item.cache_key,
			inputs: &item.inputs,
			ref_count: item.ref_count,
		})
	}
}

impl<'e> ExactSizeIterator for ComputeSequenceIter<'e> {
	fn len(&self) -> usize {
		self.postorder_iter.len()
	}
}

impl<'e> IntoIterator for &'e ComputeSequence {
	type Item = PostorderItem<'e>;
	type IntoIter = ComputeSequenceIter<'e>;

	fn into_iter(self) -> Self::IntoIter {
		ComputeSequenceIter { postorder_iter: self.postorder.iter() }
	}
}

impl ComputeSequence {
	pub fn new(expr: Rc<Expr>) -> Self {
		let e = expr.as_ref() as *const Expr;
		let mut result = ComputeSequence {
			expr,
			roots: HashMap::new(),
			postorder: Vec::new(),
			processed: HashMap::new(),
			swapped_operands: HashSet::new(),
		};

		result.find_kernel_roots(unsafe { &*e });
		result.roots.insert(e, -1);

		let mut parent_inputs = Vec::new();
		result.find_postorder(e, &mut parent_inputs);

		result
	}

	pub fn draw_expr(&self, dotfile: &str) {
		self.expr.draw(dotfile, Some(&self.roots));
	}

	pub fn item_dtype(&self, index: usize) -> DType {
		let item = &self.postorder[index];
		unsafe { (*item.expr).dtype }
	}

	pub fn len(&self) -> usize {
		self.postorder.len()
	}

	pub fn iter(&self) -> ComputeSequenceIter {
		self.into_iter()
	}

	// returns true if expr is constant
	fn find_kernel_roots(&mut self, expr: &Expr) -> bool {
		let e = expr as *const Expr;
		if let Some(is_const) = self.processed.get(&e)
			&& !is_const
		{
			self.roots.insert(e, -1);
			return false;
		}

		let is_const = match &expr.kind {
			ExprKind::Input(..) | ExprKind::Randn() => {
				self.roots.insert(e, -1);
				false
			},
			ExprKind::Const(..) => {
				// do nothing
				true
			},
			ExprKind::Unary(u) => {
				let a_const = self.find_kernel_roots(&*u.a);
				a_const
			},
			ExprKind::Binary(b) => {
				let a_const = self.find_kernel_roots(&*b.a);
				let b_const = self.find_kernel_roots(&*b.b);
				a_const && b_const
			},
			ExprKind::Reduce(r) => {
				self.find_kernel_roots(&*r.a);

				self.roots.insert(r.a.as_ref() as *const Expr, -1);
				self.roots.insert(e, -1);

				// Reduction is always marked as a root, so from the point of view of the parent,
				// it is not constant
				false
			},
			ExprKind::MatMul(m) => {
				self.find_kernel_roots(&*m.a);
				self.find_kernel_roots(&*m.b);

				// TODO - we mark both inputs as roots,
				// which will make the common idiom `x * w.T` inefficient
				self.roots.insert(m.a.as_ref() as *const Expr, -1);
				self.roots.insert(m.b.as_ref() as *const Expr, -1);
				self.roots.insert(e, -1);

				// MatMul is always marked as a root, so from the point of view of the parent,
				// it is not constant
				false
			},
			ExprKind::Transpose(t) => {
				let a_const = self.find_kernel_roots(&*t.a);
				a_const
			},
			ExprKind::Broadcast(b) => {
				// When we broadcast an expression, the calculation will be repeated several times.
				// We will do it only if the expression is constant.
				// If it's not, mark the expression as a root so that it is calculated only once.
				let a_const = self.find_kernel_roots(&*b.a);
				if !a_const {
					self.roots.insert(b.a.as_ref() as *const Expr, -1);
				}
				a_const
			},
			ExprKind::SimpleReshape(sr) => {
				let a_const = self.find_kernel_roots(&*sr.a);
				a_const
			},
		};

		self.processed.insert(e, is_const);
		is_const
	}

	fn traverse_children(&mut self, expr: &Expr, parent_inputs: &mut Vec<usize>) -> String {
		match &expr.kind {
			ExprKind::Input(..) | ExprKind::Randn() => {
				// Input and Randn are handled separately, so their cache_key is never used
				String::new()
			},
			ExprKind::Const(c) => format!("{}({})", expr.dtype, c),
			ExprKind::Unary(un) => {
				let a = self.find_postorder(&*un.a, parent_inputs);
				format!("{}({})", un.op.symbol(), a)
			},
			ExprKind::Binary(bin) => {
				let i = parent_inputs.len();
				let mut a = self.find_postorder(&*bin.a, parent_inputs);
				let j = parent_inputs.len();
				let mut b = self.find_postorder(&*bin.b, parent_inputs);
				if bin.op.is_commutative() && a > b {
					// swap a, b
					std::mem::swap(&mut a, &mut b);
					// swap inputs
					parent_inputs[i..].rotate_left(j - i);
					self.swapped_operands.insert(expr);
				}
				format!("({}{}{})", a, bin.op.symbol(), b)
			},
			ExprKind::Reduce(r) => {
				self.find_postorder(&*r.a, parent_inputs);

				// Reduce ops are handled separately, so their cache_key is never used
				String::new()
			},
			ExprKind::MatMul(m) => {
				self.find_postorder(&*m.a, parent_inputs);
				self.find_postorder(&*m.b, parent_inputs);

				// MatMuls are handled separately, so their cache_key is never used
				String::new()
			},
			ExprKind::Transpose(t) => {
				let a = self.find_postorder(&*t.a, parent_inputs);
				format!(
					"T({},{},{})",
					a,
					std::cmp::min(t.x1, t.x2),
					std::cmp::max(t.x1, t.x2)
				)
			},
			ExprKind::Broadcast(br) => {
				let a = self.find_postorder(&*br.a, parent_inputs);
				format!("B({},{},{})", a, expr.shape, br.a.shape)
			},
			ExprKind::SimpleReshape(sr) => {
				let a = self.find_postorder(&*sr.a, parent_inputs);
				format!("R({},{},{})", a, expr.shape, sr.a.shape)
			},
		}
	}

	fn find_postorder(&mut self, expr: *const Expr, parent_inputs: &mut Vec<usize>) -> String {
		let expr_ptr = expr;
		let expr = unsafe { &*expr_ptr };

		// Check if `expr` is in the `roots` set
		if let Some(entry) = self.roots.get(&expr_ptr) {
			// `expr` is a root. The value stored in `entry` is:
			// - negative if not yet processed, or
			// - an index into `postorder` if already processed
			let index = *entry;
			if index >= 0 {
				// already processed - just add `expr` as an input to the parent
				self.postorder[index as usize].ref_count += 1;
				parent_inputs.push(index as usize);
			} else {
				// not yet processed - process it
				let mut my_inputs = Vec::new();

				let cache_key = self.traverse_children(expr, &mut my_inputs);
				let cache_key = format!("{}D:{}", expr.shape.ndim(), cache_key);

				parent_inputs.push(self.postorder.len());
				self.roots.insert(expr, self.postorder.len() as isize);
				self.postorder.push(_PostorderItem {
					expr: expr_ptr,
					cache_key,
					inputs: my_inputs,
					ref_count: 1,
				});
			}

			format!("{}", expr.dtype)
		} else {
			// `expr` is NOT a root - process its children
			let cache_key = self.traverse_children(expr, parent_inputs);
			cache_key
		}
	}

	pub fn is_root(&self, expr: &Expr) -> bool {
		let expr = expr as *const Expr;
		self.roots.contains_key(&expr)
	}

	pub fn has_swapped_operands(&self, expr: &Expr) -> bool {
		let expr = expr as *const Expr;
		self.swapped_operands.contains(&expr)
	}
}
