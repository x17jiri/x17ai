// Generated file, do not edit

//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::tensor::{Tensor, TensorOpError};

use super::Kernel;
use super::builder::KernelBuilder;
use super::library::KernelLibrary;
use super::lookup::{
	AddLookupExpr,
	KernelCall,
	LnClampedLookupExpr,
	LookupWrapper,
	MulLookupExpr,
	RecipLookupExpr,
	SqrtLookupExpr,
	SubLookupExpr,
	SumLookupExpr,
	SwishLookupExpr,
};

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct RmsKernel {
	kernel: Kernel<0, 2, 1>,
}

impl RmsKernel {
	fn new() -> Self {
		let (builder, [], [a, b], [sum_to_mean]) =
			KernelBuilder::new(
				"rms", [], ["a", "b"], ["sum_to_mean"]
			);
		let kernel = builder.build(((a * b).sum() * sum_to_mean).sqrt());
		Self { kernel }
	}
}

type RmsExpr<'a> =
	SqrtLookupExpr<
		MulLookupExpr<
			SumLookupExpr<
				MulLookupExpr<
					&'a Tensor,
					&'a Tensor,
				>,
			>,
			f64,
		>,
	>;

impl<'a> KernelCall<RmsExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<RmsExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let SqrtLookupExpr(
			MulLookupExpr(
				SumLookupExpr(
					MulLookupExpr(
						a,
						b,
					),
				),
				sum_to_mean,
			),
		) = expr.0;
		self.data.rms.kernel.run(to, [], [a, b], [sum_to_mean])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct RmsRecipKernel {
	kernel: Kernel<0, 2, 2>,
}

impl RmsRecipKernel {
	fn new() -> Self {
		let (builder, [], [a, b], [eps, sum_to_mean]) =
			KernelBuilder::new(
				"rms_recip", [], ["a", "b"], ["eps", "sum_to_mean"]
			);
		let kernel = builder.build(((a * b).sum() * sum_to_mean).sqrt().recip(eps));
		Self { kernel }
	}
}

type RmsRecipExpr<'a> =
	RecipLookupExpr<
		SqrtLookupExpr<
			MulLookupExpr<
				SumLookupExpr<
					MulLookupExpr<
						&'a Tensor,
						&'a Tensor,
					>,
				>,
				f64,
			>,
		>,
		f64,
	>;

impl<'a> KernelCall<RmsRecipExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<RmsRecipExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let RecipLookupExpr(
			SqrtLookupExpr(
				MulLookupExpr(
					SumLookupExpr(
						MulLookupExpr(
							a,
							b,
						),
					),
					sum_to_mean,
				),
			),
			eps,
		) = expr.0;
		self.data.rms_recip.kernel.run(to, [], [a, b], [eps, sum_to_mean])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct AddKernel {
	kernel: Kernel<2, 0, 0>,
}

impl AddKernel {
	fn new() -> Self {
		let (builder, [a, b], [], []) =
			KernelBuilder::new(
				"add", ["a", "b"], [], []
			);
		let kernel = builder.build(a + b);
		Self { kernel }
	}
}

type AddExpr<'a> =
	AddLookupExpr<
		&'a Tensor,
		&'a Tensor,
	>;

impl<'a> KernelCall<AddExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<AddExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let AddLookupExpr(
			a,
			b,
		) = expr.0;
		self.data.add.kernel.run(to, [a, b], [], [])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct SubKernel {
	kernel: Kernel<2, 0, 0>,
}

impl SubKernel {
	fn new() -> Self {
		let (builder, [a, b], [], []) =
			KernelBuilder::new(
				"sub", ["a", "b"], [], []
			);
		let kernel = builder.build(a - b);
		Self { kernel }
	}
}

type SubExpr<'a> =
	SubLookupExpr<
		&'a Tensor,
		&'a Tensor,
	>;

impl<'a> KernelCall<SubExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<SubExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let SubLookupExpr(
			a,
			b,
		) = expr.0;
		self.data.sub.kernel.run(to, [a, b], [], [])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct MulKernel {
	kernel: Kernel<2, 0, 0>,
}

impl MulKernel {
	fn new() -> Self {
		let (builder, [a, b], [], []) =
			KernelBuilder::new(
				"mul", ["a", "b"], [], []
			);
		let kernel = builder.build(a * b);
		Self { kernel }
	}
}

type MulExpr<'a> =
	MulLookupExpr<
		&'a Tensor,
		&'a Tensor,
	>;

impl<'a> KernelCall<MulExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<MulExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			a,
			b,
		) = expr.0;
		self.data.mul.kernel.run(to, [a, b], [], [])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct AccMulKernel {
	kernel: Kernel<3, 0, 0>,
}

impl AccMulKernel {
	fn new() -> Self {
		let (builder, [x, a, b], [], []) =
			KernelBuilder::new(
				"acc_mul", ["x", "a", "b"], [], []
			);
		let kernel = builder.build(x + (a * b));
		Self { kernel }
	}
}

type AccMulExpr<'a> =
	AddLookupExpr<
		&'a Tensor,
		MulLookupExpr<
			&'a Tensor,
			&'a Tensor,
		>,
	>;

impl<'a> KernelCall<AccMulExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<AccMulExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let AddLookupExpr(
			x,
			MulLookupExpr(
				a,
				b,
			),
		) = expr.0;
		self.data.acc_mul.kernel.run(to, [x, a, b], [], [])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct MulScaledKernel {
	kernel: Kernel<2, 0, 1>,
}

impl MulScaledKernel {
	fn new() -> Self {
		let (builder, [a, b], [], [scale]) =
			KernelBuilder::new(
				"mul_scaled", ["a", "b"], [], ["scale"]
			);
		let kernel = builder.build((a * b) * scale);
		Self { kernel }
	}
}

type MulScaledExpr<'a> =
	MulLookupExpr<
		MulLookupExpr<
			&'a Tensor,
			&'a Tensor,
		>,
		f64,
	>;

impl<'a> KernelCall<MulScaledExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<MulScaledExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			MulLookupExpr(
				a,
				b,
			),
			scale,
		) = expr.0;
		self.data.mul_scaled.kernel.run(to, [a, b], [], [scale])
	}
}

//--------------------------------------------------------------------------------------------------

type MulScaled2Expr<'a> =
	MulLookupExpr<
		MulLookupExpr<
			MulLookupExpr<
				&'a Tensor,
				&'a Tensor,
			>,
			f64,
		>,
		f64,
	>;

impl<'a> KernelCall<MulScaled2Expr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<MulScaled2Expr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			MulLookupExpr(
				MulLookupExpr(
					a,
					b,
				),
				scale1,
			),
			scale2,
		) = expr.0;
		self.data.mul_scaled2.kernel.run(to, [a, b], [], [scale1, scale2])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct MulXLnYKernel {
	kernel: Kernel<2, 0, 0>,
}

impl MulXLnYKernel {
	fn new() -> Self {
		let (builder, [x, y], [], []) =
			KernelBuilder::new(
				"mul_x_ln_y", ["x", "y"], [], []
			);
		let kernel = builder.build(x * y.ln_clamped());
		Self { kernel }
	}
}

type MulXLnYExpr<'a> =
	MulLookupExpr<
		&'a Tensor,
		LnClampedLookupExpr<
			&'a Tensor,
		>,
	>;

impl<'a> KernelCall<MulXLnYExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<MulXLnYExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			x,
			LnClampedLookupExpr(
				y,
			),
		) = expr.0;
		self.data.mul_x_ln_y.kernel.run(to, [x, y], [], [])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct WeightedAddKernel {
	kernel: Kernel<2, 0, 2>,
}

impl WeightedAddKernel {
	fn new() -> Self {
		let (builder, [a, b], [], [a_weight, b_weight]) =
			KernelBuilder::new(
				"weighted_add", ["a", "b"], [], ["a_weight", "b_weight"]
			);
		let kernel = builder.build((a * a_weight) + (b * b_weight));
		Self { kernel }
	}
}

type WeightedAddExpr<'a> =
	AddLookupExpr<
		MulLookupExpr<
			&'a Tensor,
			f64,
		>,
		MulLookupExpr<
			&'a Tensor,
			f64,
		>,
	>;

impl<'a> KernelCall<WeightedAddExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<WeightedAddExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let AddLookupExpr(
			MulLookupExpr(
				a,
				a_weight,
			),
			MulLookupExpr(
				b,
				b_weight,
			),
		) = expr.0;
		self.data.weighted_add.kernel.run(to, [a, b], [], [a_weight, b_weight])
	}
}

//--------------------------------------------------------------------------------------------------

type WeightedSubExpr<'a> =
	SubLookupExpr<
		MulLookupExpr<
			&'a Tensor,
			f64,
		>,
		MulLookupExpr<
			&'a Tensor,
			f64,
		>,
	>;

impl<'a> KernelCall<WeightedSubExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<WeightedSubExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let SubLookupExpr(
			MulLookupExpr(
				a,
				a_weight,
			),
			MulLookupExpr(
				b,
				b_weight,
			),
		) = expr.0;
		self.data.weighted_sub.kernel.run(to, [a, b], [], [a_weight, b_weight])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct AddXMulScaledKernel {
	kernel: Kernel<3, 0, 1>,
}

impl AddXMulScaledKernel {
	fn new() -> Self {
		let (builder, [x, a, b], [], [scale]) =
			KernelBuilder::new(
				"add_x_mul_scaled", ["x", "a", "b"], [], ["scale"]
			);
		let kernel = builder.build((x + (a * b)) * scale);
		Self { kernel }
	}
}

type AddXMulScaledExpr<'a> =
	MulLookupExpr<
		AddLookupExpr<
			&'a Tensor,
			MulLookupExpr<
				&'a Tensor,
				&'a Tensor,
			>,
		>,
		f64,
	>;

impl<'a> KernelCall<AddXMulScaledExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<AddXMulScaledExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			AddLookupExpr(
				x,
				MulLookupExpr(
					a,
					b,
				),
			),
			scale,
		) = expr.0;
		self.data.add_x_mul_scaled.kernel.run(to, [x, a, b], [], [scale])
	}
}

//--------------------------------------------------------------------------------------------------

type AddXMulScaled2Expr<'a> =
	MulLookupExpr<
		MulLookupExpr<
			AddLookupExpr<
				&'a Tensor,
				MulLookupExpr<
					&'a Tensor,
					&'a Tensor,
				>,
			>,
			f64,
		>,
		f64,
	>;

impl<'a> KernelCall<AddXMulScaled2Expr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<AddXMulScaled2Expr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			MulLookupExpr(
				AddLookupExpr(
					x,
					MulLookupExpr(
						a,
						b,
					),
				),
				scale1,
			),
			scale2,
		) = expr.0;
		self.data.add_x_mul_scaled2.kernel.run(to, [x, a, b], [], [scale1, scale2])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct DotKernel {
	kernel: Kernel<0, 2, 0>,
}

impl DotKernel {
	fn new() -> Self {
		let (builder, [], [a, b], []) =
			KernelBuilder::new(
				"dot", [], ["a", "b"], []
			);
		let kernel = builder.build((a * b).sum());
		Self { kernel }
	}
}

type DotExpr<'a> =
	SumLookupExpr<
		MulLookupExpr<
			&'a Tensor,
			&'a Tensor,
		>,
	>;

impl<'a> KernelCall<DotExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<DotExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let SumLookupExpr(
			MulLookupExpr(
				a,
				b,
			),
		) = expr.0;
		self.data.dot.kernel.run(to, [], [a, b], [])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct DotScaledKernel {
	kernel: Kernel<0, 2, 1>,
}

impl DotScaledKernel {
	fn new() -> Self {
		let (builder, [], [a, b], [scale]) =
			KernelBuilder::new(
				"dot_scaled", [], ["a", "b"], ["scale"]
			);
		let kernel = builder.build((a * b).sum() * scale);
		Self { kernel }
	}
}

type DotScaledExpr<'a> =
	MulLookupExpr<
		SumLookupExpr<
			MulLookupExpr<
				&'a Tensor,
				&'a Tensor,
			>,
		>,
		f64,
	>;

impl<'a> KernelCall<DotScaledExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<DotScaledExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			SumLookupExpr(
				MulLookupExpr(
					a,
					b,
				),
			),
			scale,
		) = expr.0;
		self.data.dot_scaled.kernel.run(to, [], [a, b], [scale])
	}
}

//--------------------------------------------------------------------------------------------------

type DotScaled2Expr<'a> =
	MulLookupExpr<
		MulLookupExpr<
			SumLookupExpr<
				MulLookupExpr<
					&'a Tensor,
					&'a Tensor,
				>,
			>,
			f64,
		>,
		f64,
	>;

impl<'a> KernelCall<DotScaled2Expr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<DotScaled2Expr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			MulLookupExpr(
				SumLookupExpr(
					MulLookupExpr(
						a,
						b,
					),
				),
				scale1,
			),
			scale2,
		) = expr.0;
		self.data.dot_scaled2.kernel.run(to, [], [a, b], [scale1, scale2])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct WeightedAddTDotKernel {
	kernel: Kernel<1, 2, 2>,
}

impl WeightedAddTDotKernel {
	fn new() -> Self {
		let (builder, [t], [a, b], [t_weight, ab_weight]) =
			KernelBuilder::new(
				"weighted_add_t_dot", ["t"], ["a", "b"], ["t_weight", "ab_weight"]
			);
		let kernel = builder.build((t * t_weight) + ((a * b).sum() * ab_weight));
		Self { kernel }
	}
}

type WeightedAddTDotExpr<'a> =
	AddLookupExpr<
		MulLookupExpr<
			&'a Tensor,
			f64,
		>,
		MulLookupExpr<
			SumLookupExpr<
				MulLookupExpr<
					&'a Tensor,
					&'a Tensor,
				>,
			>,
			f64,
		>,
	>;

impl<'a> KernelCall<WeightedAddTDotExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<WeightedAddTDotExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let AddLookupExpr(
			MulLookupExpr(
				t,
				t_weight,
			),
			MulLookupExpr(
				SumLookupExpr(
					MulLookupExpr(
						a,
						b,
					),
				),
				ab_weight,
			),
		) = expr.0;
		self.data.weighted_add_t_dot.kernel.run(to, [t], [a, b], [t_weight, ab_weight])
	}
}

//--------------------------------------------------------------------------------------------------

type WeightedAddTDot2Expr<'a> =
	AddLookupExpr<
		MulLookupExpr<
			&'a Tensor,
			f64,
		>,
		MulLookupExpr<
			MulLookupExpr<
				SumLookupExpr<
					MulLookupExpr<
						&'a Tensor,
						&'a Tensor,
					>,
				>,
				f64,
			>,
			f64,
		>,
	>;

impl<'a> KernelCall<WeightedAddTDot2Expr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<WeightedAddTDot2Expr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let AddLookupExpr(
			MulLookupExpr(
				t,
				t_weight,
			),
			MulLookupExpr(
				MulLookupExpr(
					SumLookupExpr(
						MulLookupExpr(
							a,
							b,
						),
					),
					ab_weight1,
				),
				ab_weight2,
			),
		) = expr.0;
		self.data.weighted_add_t_dot2.kernel.run(to, [t], [a, b], [t_weight, ab_weight1, ab_weight2])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct MulSubAMulBCDKernel {
	kernel: Kernel<4, 0, 0>,
}

impl MulSubAMulBCDKernel {
	fn new() -> Self {
		let (builder, [a, b, c, d], [], []) =
			KernelBuilder::new(
				"mul_sub_a_mul_b_c_d", ["a", "b", "c", "d"], [], []
			);
		let kernel = builder.build((a - (b * c)) * d);
		Self { kernel }
	}
}

type MulSubAMulBCDExpr<'a> =
	MulLookupExpr<
		SubLookupExpr<
			&'a Tensor,
			MulLookupExpr<
				&'a Tensor,
				&'a Tensor,
			>,
		>,
		&'a Tensor,
	>;

impl<'a> KernelCall<MulSubAMulBCDExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<MulSubAMulBCDExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			SubLookupExpr(
				a,
				MulLookupExpr(
					b,
					c,
				),
			),
			d,
		) = expr.0;
		self.data.mul_sub_a_mul_b_c_d.kernel.run(to, [a, b, c, d], [], [])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct MulSubABCKernel {
	kernel: Kernel<3, 0, 0>,
}

impl MulSubABCKernel {
	fn new() -> Self {
		let (builder, [a, b, c], [], []) =
			KernelBuilder::new(
				"mul_sub_a_b_c", ["a", "b", "c"], [], []
			);
		let kernel = builder.build((a - b) * c);
		Self { kernel }
	}
}

type MulSubABCExpr<'a> =
	MulLookupExpr<
		SubLookupExpr<
			&'a Tensor,
			&'a Tensor,
		>,
		&'a Tensor,
	>;

impl<'a> KernelCall<MulSubABCExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<MulSubABCExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			SubLookupExpr(
				a,
				b,
			),
			c,
		) = expr.0;
		self.data.mul_sub_a_b_c.kernel.run(to, [a, b, c], [], [])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct SqrtRecipKernel {
	kernel: Kernel<1, 0, 1>,
}

impl SqrtRecipKernel {
	fn new() -> Self {
		let (builder, [a], [], [eps]) =
			KernelBuilder::new(
				"sqrt_recip", ["a"], [], ["eps"]
			);
		let kernel = builder.build(a.sqrt().recip(eps));
		Self { kernel }
	}
}

type SqrtRecipExpr<'a> =
	RecipLookupExpr<
		SqrtLookupExpr<
			&'a Tensor,
		>,
		f64,
	>;

impl<'a> KernelCall<SqrtRecipExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<SqrtRecipExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let RecipLookupExpr(
			SqrtLookupExpr(
				a,
			),
			eps,
		) = expr.0;
		self.data.sqrt_recip.kernel.run(to, [a], [], [eps])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct AccMulScaledKernel {
	kernel: Kernel<3, 0, 1>,
}

impl AccMulScaledKernel {
	fn new() -> Self {
		let (builder, [x, a, b], [], [scale]) =
			KernelBuilder::new(
				"acc_mul_scaled", ["x", "a", "b"], [], ["scale"]
			);
		let kernel = builder.build(x + (a * b * scale));
		Self { kernel }
	}
}

type AccMulScaledExpr<'a> =
	AddLookupExpr<
		&'a Tensor,
		MulLookupExpr<
			MulLookupExpr<
				&'a Tensor,
				&'a Tensor,
			>,
			f64,
		>,
	>;

impl<'a> KernelCall<AccMulScaledExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<AccMulScaledExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let AddLookupExpr(
			x,
			MulLookupExpr(
				MulLookupExpr(
					a,
					b,
				),
				scale,
			),
		) = expr.0;
		self.data.acc_mul_scaled.kernel.run(to, [x, a, b], [], [scale])
	}
}

//--------------------------------------------------------------------------------------------------

type AccNegMulScaledExpr<'a> =
	SubLookupExpr<
		&'a Tensor,
		MulLookupExpr<
			MulLookupExpr<
				&'a Tensor,
				&'a Tensor,
			>,
			f64,
		>,
	>;

impl<'a> KernelCall<AccNegMulScaledExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<AccNegMulScaledExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let SubLookupExpr(
			x,
			MulLookupExpr(
				MulLookupExpr(
					a,
					b,
				),
				scale,
			),
		) = expr.0;
		self.data.acc_neg_mul_scaled.kernel.run(to, [x, a, b], [], [scale])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct SwigluKernel {
	kernel: Kernel<2, 0, 0>,
}

impl SwigluKernel {
	fn new() -> Self {
		let (builder, [lin, gate], [], []) =
			KernelBuilder::new(
				"swiglu", ["lin", "gate"], [], []
			);
		let kernel = builder.build(lin * gate.swish());
		Self { kernel }
	}
}

type SwigluExpr<'a> =
	MulLookupExpr<
		&'a Tensor,
		SwishLookupExpr<
			&'a Tensor,
		>,
	>;

impl<'a> KernelCall<SwigluExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<SwigluExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let MulLookupExpr(
			lin,
			SwishLookupExpr(
				gate,
			),
		) = expr.0;
		self.data.swiglu.kernel.run(to, [lin, gate], [], [])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct FillKernel {
	kernel: Kernel<0, 0, 1>,
}

impl FillKernel {
	fn new() -> Self {
		let (builder, [], [], [v]) =
			KernelBuilder::new(
				"fill", [], [], ["v"]
			);
		let kernel = builder.build(v);
		Self { kernel }
	}
}

type FillExpr<'a> =
	f64;

impl<'a> KernelCall<FillExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<FillExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let v = expr.0;
		self.data.fill.kernel.run(to, [], [], [v])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct CopyKernel {
	kernel: Kernel<1, 0, 0>,
}

impl CopyKernel {
	fn new() -> Self {
		let (builder, [v], [], []) =
			KernelBuilder::new(
				"copy", ["v"], [], []
			);
		let kernel = builder.build(v);
		Self { kernel }
	}
}

type CopyExpr<'a> =
	&'a Tensor;

impl<'a> KernelCall<CopyExpr<'a>> for KernelLibrary {
	fn call(&self, to: &Tensor, expr: LookupWrapper<CopyExpr<'a>>) -> Result<(), ErrPack<TensorOpError>> {
		let v = expr.0;
		self.data.copy.kernel.run(to, [v], [], [])
	}
}

//--------------------------------------------------------------------------------------------------

pub struct KernelLibraryData {
	rms: RmsKernel,
	rms_recip: RmsRecipKernel,
	add: AddKernel,
	sub: SubKernel,
	mul: MulKernel,
	acc_mul: AccMulKernel,
	mul_scaled: MulScaledKernel,
	mul_x_ln_y: MulXLnYKernel,
	weighted_add: WeightedAddKernel,
	add_x_mul_scaled: AddXMulScaledKernel,
	dot: DotKernel,
	dot_scaled: DotScaledKernel,
	weighted_add_t_dot: WeightedAddTDotKernel,
	mul_sub_a_mul_b_c_d: MulSubAMulBCDKernel,
	mul_sub_a_b_c: MulSubABCKernel,
	sqrt_recip: SqrtRecipKernel,
	acc_mul_scaled: AccMulScaledKernel,
	swiglu: SwigluKernel,
	fill: FillKernel,
	copy: CopyKernel,
}

impl KernelLibraryData {
	pub fn new() -> Self {
		Self {
			rms: RmsKernel::new(),
			rms_recip: RmsRecipKernel::new(),
			add: AddKernel::new(),
			sub: SubKernel::new(),
			mul: MulKernel::new(),
			acc_mul: AccMulKernel::new(),
			mul_scaled: MulScaledKernel::new(),
			mul_x_ln_y: MulXLnYKernel::new(),
			weighted_add: WeightedAddKernel::new(),
			add_x_mul_scaled: AddXMulScaledKernel::new(),
			dot: DotKernel::new(),
			dot_scaled: DotScaledKernel::new(),
			weighted_add_t_dot: WeightedAddTDotKernel::new(),
			mul_sub_a_mul_b_c_d: MulSubAMulBCDKernel::new(),
			mul_sub_a_b_c: MulSubABCKernel::new(),
			sqrt_recip: SqrtRecipKernel::new(),
			acc_mul_scaled: AccMulScaledKernel::new(),
			swiglu: SwigluKernel::new(),
			fill: FillKernel::new(),
			copy: CopyKernel::new(),
		}
	}
}

//--------------------------------------------------------------------------------------------------
