//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::rc::Rc;

use crate::util::array::map_into;

//--------------------------------------------------------------------------------------------------

pub enum ScalarExprData {
	ElemArg(Rc<ElemArgData>),
	ConstArg(Rc<ConstArgData>),
	FloatLiteral(FloatLiteral),
	Dot(Dot),
}

pub struct ScalarExpr {
	pub data: Rc<ScalarExprData>,
}

//--------------------------------------------------------------------------------------------------

pub struct ElemArgData {
	pub index: usize,
	pub name: String,
}

pub struct ConstArgData {
	pub index: usize,
	pub name: String,
}

pub struct FloatLiteral {
	pub value: f64,
}

pub struct VecArgData {
	pub index: usize,
	pub name: String,
}

pub struct VecArg {
	pub data: Rc<VecArgData>,
}

pub struct Kernel {
	pub elem_args: Vec<Rc<ElemArgData>>,
	pub vec_args: Vec<Rc<VecArgData>>,
	pub const_args: Vec<Rc<ConstArgData>>,
	pub expr: ScalarExpr,
}

pub struct KernelBuilder {
	elem_args: Vec<Rc<ElemArgData>>,
	vec_args: Vec<Rc<VecArgData>>,
	const_args: Vec<Rc<ConstArgData>>,
}

impl KernelBuilder {
	pub fn new<const E: usize, const V: usize, const C: usize>(
		elem_args: [&str; E],
		vec_args: [&str; V],
		const_args: [&str; C],
	) -> (Self, [ScalarExpr; E], [VecArg; V], [ScalarExpr; C]) {
		let elem_args_data = map_into(elem_args, |index, name| {
			//
			Rc::new(ElemArgData { index, name: name.to_string() })
		});
		let vec_args_data = map_into(vec_args, |index, name| {
			//
			Rc::new(VecArgData { index, name: name.to_string() })
		});
		let const_args_data = map_into(const_args, |index, name| {
			//
			Rc::new(ConstArgData { index, name: name.to_string() })
		});

		let builder = Self {
			elem_args: elem_args_data.to_vec(),
			vec_args: vec_args_data.to_vec(),
			const_args: const_args_data.to_vec(),
		};

		let elem_args_exprs = elem_args_data.map(|data| ScalarExpr {
			data: Rc::new(ScalarExprData::ElemArg(data)),
		});
		let vec_args_exprs = vec_args_data.map(|data| VecArg { data });
		let const_args_exprs = const_args_data.map(|data| ScalarExpr {
			data: Rc::new(ScalarExprData::ConstArg(data)),
		});

		(builder, elem_args_exprs, vec_args_exprs, const_args_exprs)
	}

	pub fn build(self, expr: ScalarExpr) -> Kernel {
		Kernel {
			elem_args: self.elem_args,
			vec_args: self.vec_args,
			const_args: self.const_args,
			expr,
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct VecMul {
	a: Rc<VecArgData>,
	b: Rc<VecArgData>,
}

impl VecMul {
	pub fn sum(self) -> Rc<Dot> {
		Rc::new(Dot { a: self.a, b: self.b })
	}
}

impl std::ops::Mul<Self> for VecArg {
	type Output = VecMul;

	fn mul(self, rhs: Self) -> Self::Output {
		VecMul { a: self.data, b: rhs.data }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Dot {
	pub a: Rc<VecArgData>,
	pub b: Rc<VecArgData>,
}

//--------------------------------------------------------------------------------------------------
