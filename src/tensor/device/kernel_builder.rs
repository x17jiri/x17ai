//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::Arc;

use crate::tensor::device::kernel::{ConstArg, ElemArg, Kernel, KernelData, ReduceArg, ScalarExpr};
use crate::tensor::device::kernel_registry::KernelRegistry;
use crate::util::array;

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct ScalarExprWrapper {
	pub expr: Arc<ScalarExpr>,
}

#[derive(Clone)]
pub struct ReduceArgWrapper {
	pub arg: Arc<ReduceArg>,
}

//--------------------------------------------------------------------------------------------------

pub struct KernelBuilder<const E: usize, const V: usize, const C: usize> {
	pub(crate) name: String,
	pub(crate) elem_args: [Arc<ElemArg>; E],
	pub(crate) reduce_args: [Arc<ReduceArg>; V],
	pub(crate) const_args: [Arc<ConstArg>; C],
}

impl<const E: usize, const V: usize, const C: usize> KernelBuilder<E, V, C> {
	pub fn new(
		name: &str,
		elem_args: [&str; E],
		reduce_args: [&str; V],
		const_args: [&str; C],
	) -> (Self, [ScalarExprWrapper; E], [ReduceArgWrapper; V], [ScalarExprWrapper; C]) {
		let elem_args = array::map_into(elem_args, |index, name| {
			Arc::new(ElemArg { index, name: name.to_string() })
		});
		let reduce_args = array::map_into(reduce_args, |index, name| {
			Arc::new(ReduceArg { index, name: name.to_string() })
		});
		let const_args = array::map_into(const_args, |index, name| {
			Arc::new(ConstArg { index, name: name.to_string() })
		});

		let builder = Self {
			name: name.to_string(),
			elem_args: elem_args.clone(),
			reduce_args: reduce_args.clone(),
			const_args: const_args.clone(),
		};

		let elem_args_exprs =
			elem_args.map(|a| ScalarExprWrapper { expr: Arc::new(ScalarExpr::ElemArg(a)) });
		let reduce_args_exprs = reduce_args.map(|a| ReduceArgWrapper { arg: a });
		let const_args_exprs =
			const_args.map(|a| ScalarExprWrapper { expr: Arc::new(ScalarExpr::ConstArg(a)) });

		(builder, elem_args_exprs, reduce_args_exprs, const_args_exprs)
	}

	pub fn build(self, expr: ScalarExprWrapper) -> Kernel<E, V, C> {
		let ScalarExprWrapper { expr } = expr;
		let Self { name, elem_args, reduce_args, const_args } = self;

		let elem_args = elem_args.into();
		let reduce_args = reduce_args.into();
		let const_args = const_args.into();

		let reg = KernelRegistry::instance();
		let mut reg = reg.write().unwrap();
		let data = reg.add_kernel(|id| {
			Arc::new(KernelData {
				id,
				name,
				elem_args,
				reduce_args,
				const_args,
				expr,
			})
		});
		Kernel::new(data)
	}
}

//--------------------------------------------------------------------------------------------------

impl std::ops::Add<Self> for ScalarExprWrapper {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self {
			expr: Arc::new(ScalarExpr::AddExpr(self.expr, rhs.expr)),
		}
	}
}

impl std::ops::Mul<Self> for ScalarExprWrapper {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Self {
			expr: Arc::new(ScalarExpr::MulExpr(self.expr, rhs.expr)),
		}
	}
}

impl ScalarExprWrapper {
	pub fn sqrt(self) -> Self {
		Self {
			expr: Arc::new(ScalarExpr::SqrtExpr(self.expr)),
		}
	}

	pub fn recip(self, eps: ScalarExprWrapper) -> Self {
		Self {
			expr: Arc::new(ScalarExpr::RecipExpr(self.expr, eps.expr)),
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct ReduceMul(Arc<ReduceArg>, Arc<ReduceArg>);

impl ReduceMul {
	pub fn sum(self) -> ScalarExprWrapper {
		ScalarExprWrapper {
			expr: Arc::new(ScalarExpr::DotExpr(self.0, self.1)),
		}
	}
}

impl std::ops::Mul<Self> for ReduceArgWrapper {
	type Output = ReduceMul;

	fn mul(self, rhs: Self) -> Self::Output {
		ReduceMul(self.arg, rhs.arg)
	}
}

//--------------------------------------------------------------------------------------------------
