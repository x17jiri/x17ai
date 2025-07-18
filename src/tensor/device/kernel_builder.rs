//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::Arc;

use crate::tensor::device::kernel::{ConstArg, ElemArg, Kernel, KernelData, ScalarExpr, VecArg};
use crate::tensor::device::kernel_registry::KernelRegistry;
use crate::util::array;

//--------------------------------------------------------------------------------------------------

pub struct ScalarExprWrapper {
	pub expr: Arc<ScalarExpr>,
}

pub struct VecArgWrapper {
	pub arg: Arc<VecArg>,
}

//--------------------------------------------------------------------------------------------------

pub struct KernelBuilder<const E: usize, const V: usize, const C: usize> {
	pub(crate) name: String,
	pub(crate) elem_args: [Arc<ElemArg>; E],
	pub(crate) vec_args: [Arc<VecArg>; V],
	pub(crate) const_args: [Arc<ConstArg>; C],
}

impl<const E: usize, const V: usize, const C: usize> KernelBuilder<E, V, C> {
	pub fn new(
		name: &str,
		elem_args: [&str; E],
		vec_args: [&str; V],
		const_args: [&str; C],
	) -> (Self, [ScalarExprWrapper; E], [VecArgWrapper; V], [ScalarExprWrapper; C]) {
		let elem_args = array::map_into(elem_args, |index, name| {
			Arc::new(ElemArg { index, name: name.to_string() })
		});
		let vec_args = array::map_into(vec_args, |index, name| {
			Arc::new(VecArg { index, name: name.to_string() })
		});
		let const_args = array::map_into(const_args, |index, name| {
			Arc::new(ConstArg { index, name: name.to_string() })
		});

		let builder = Self {
			name: name.to_string(),
			elem_args: elem_args.clone(),
			vec_args: vec_args.clone(),
			const_args: const_args.clone(),
		};

		let elem_args_exprs =
			elem_args.map(|a| ScalarExprWrapper { expr: Arc::new(ScalarExpr::ElemArg(a)) });
		let vec_args_exprs = vec_args.map(|a| VecArgWrapper { arg: a });
		let const_args_exprs =
			const_args.map(|a| ScalarExprWrapper { expr: Arc::new(ScalarExpr::ConstArg(a)) });

		(builder, elem_args_exprs, vec_args_exprs, const_args_exprs)
	}

	pub fn build(self, expr: ScalarExprWrapper) -> Kernel<E, V, C> {
		let ScalarExprWrapper { expr } = expr;
		let Self { name, elem_args, vec_args, const_args } = self;

		let elem_args = elem_args.into();
		let vec_args = vec_args.into();
		let const_args = const_args.into();

		let reg = KernelRegistry::instance();
		let mut reg = reg.write().unwrap();
		let data = reg.add_kernel(|id| {
			Arc::new(KernelData {
				id,
				name,
				elem_args,
				vec_args,
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

//--------------------------------------------------------------------------------------------------

pub struct VecMul(Arc<VecArg>, Arc<VecArg>);

impl VecMul {
	pub fn sum(self) -> ScalarExprWrapper {
		ScalarExprWrapper {
			expr: Arc::new(ScalarExpr::DotExpr(self.0, self.1)),
		}
	}
}

impl std::ops::Mul<Self> for VecArgWrapper {
	type Output = VecMul;

	fn mul(self, rhs: Self) -> Self::Output {
		VecMul(self.arg, rhs.arg)
	}
}

//--------------------------------------------------------------------------------------------------
