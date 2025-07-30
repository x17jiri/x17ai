// Generated file, do not edit

//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::tensor::math::EvaluatesToTensor;
use crate::tensor::{Tensor, TensorOpError};

use super::Kernel;
use super::builder::KernelBuilder;
use super::library::KernelLibrary;
use super::lookup::{
	AddLookupExpr,
	KernelLookup,
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor, sum_to_mean: f64) -> RmsKernelCall<'a> {
		RmsKernelCall { kernel: self, a, b, sum_to_mean }
	}
}

pub struct RmsKernelCall<'a> {
	kernel: &'a RmsKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	sum_to_mean: f64,
}

impl<'a> EvaluatesToTensor for RmsKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [], [self.a, self.b], [self.sum_to_mean])
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

impl<'a> KernelLookup<RmsExpr<'a>> for KernelLibrary {
	type CallType = RmsKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<RmsExpr<'a>>) -> RmsKernelCall<'a> {
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
		self.data.rms.call(a, b, sum_to_mean)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor, eps: f64, sum_to_mean: f64) -> RmsRecipKernelCall<'a> {
		RmsRecipKernelCall { kernel: self, a, b, eps, sum_to_mean }
	}
}

pub struct RmsRecipKernelCall<'a> {
	kernel: &'a RmsRecipKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	eps: f64,
	sum_to_mean: f64,
}

impl<'a> EvaluatesToTensor for RmsRecipKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [], [self.a, self.b], [self.eps, self.sum_to_mean])
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

impl<'a> KernelLookup<RmsRecipExpr<'a>> for KernelLibrary {
	type CallType = RmsRecipKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<RmsRecipExpr<'a>>) -> RmsRecipKernelCall<'a> {
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
		self.data.rms_recip.call(a, b, eps, sum_to_mean)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor) -> AddKernelCall<'a> {
		AddKernelCall { kernel: self, a, b }
	}
}

pub struct AddKernelCall<'a> {
	kernel: &'a AddKernel,
	a: &'a Tensor,
	b: &'a Tensor,
}

impl<'a> EvaluatesToTensor for AddKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b], [], [])
	}
}

type AddExpr<'a> =
	AddLookupExpr<
		&'a Tensor,
		&'a Tensor,
	>;

impl<'a> KernelLookup<AddExpr<'a>> for KernelLibrary {
	type CallType = AddKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<AddExpr<'a>>) -> AddKernelCall<'a> {
		let AddLookupExpr(
			a,
			b,
		) = expr.0;
		self.data.add.call(a, b)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor) -> SubKernelCall<'a> {
		SubKernelCall { kernel: self, a, b }
	}
}

pub struct SubKernelCall<'a> {
	kernel: &'a SubKernel,
	a: &'a Tensor,
	b: &'a Tensor,
}

impl<'a> EvaluatesToTensor for SubKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b], [], [])
	}
}

type SubExpr<'a> =
	SubLookupExpr<
		&'a Tensor,
		&'a Tensor,
	>;

impl<'a> KernelLookup<SubExpr<'a>> for KernelLibrary {
	type CallType = SubKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<SubExpr<'a>>) -> SubKernelCall<'a> {
		let SubLookupExpr(
			a,
			b,
		) = expr.0;
		self.data.sub.call(a, b)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor) -> MulKernelCall<'a> {
		MulKernelCall { kernel: self, a, b }
	}
}

pub struct MulKernelCall<'a> {
	kernel: &'a MulKernel,
	a: &'a Tensor,
	b: &'a Tensor,
}

impl<'a> EvaluatesToTensor for MulKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b], [], [])
	}
}

type MulExpr<'a> =
	MulLookupExpr<
		&'a Tensor,
		&'a Tensor,
	>;

impl<'a> KernelLookup<MulExpr<'a>> for KernelLibrary {
	type CallType = MulKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulExpr<'a>>) -> MulKernelCall<'a> {
		let MulLookupExpr(
			a,
			b,
		) = expr.0;
		self.data.mul.call(a, b)
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

	pub fn call<'a>(&'a self, x: &'a Tensor, a: &'a Tensor, b: &'a Tensor) -> AccMulKernelCall<'a> {
		AccMulKernelCall { kernel: self, x, a, b }
	}
}

pub struct AccMulKernelCall<'a> {
	kernel: &'a AccMulKernel,
	x: &'a Tensor,
	a: &'a Tensor,
	b: &'a Tensor,
}

impl<'a> EvaluatesToTensor for AccMulKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.x, self.a, self.b], [], [])
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

impl<'a> KernelLookup<AccMulExpr<'a>> for KernelLibrary {
	type CallType = AccMulKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<AccMulExpr<'a>>) -> AccMulKernelCall<'a> {
		let AddLookupExpr(
			x,
			MulLookupExpr(
				a,
				b,
			),
		) = expr.0;
		self.data.acc_mul.call(x, a, b)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor, scale: f64) -> MulScaledKernelCall<'a> {
		MulScaledKernelCall { kernel: self, a, b, scale }
	}
}

pub struct MulScaledKernelCall<'a> {
	kernel: &'a MulScaledKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	scale: f64,
}

impl<'a> EvaluatesToTensor for MulScaledKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b], [], [self.scale])
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

impl<'a> KernelLookup<MulScaledExpr<'a>> for KernelLibrary {
	type CallType = MulScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulScaledExpr<'a>>) -> MulScaledKernelCall<'a> {
		let MulLookupExpr(
			MulLookupExpr(
				a,
				b,
			),
			scale,
		) = expr.0;
		self.data.mul_scaled.call(a, b, scale)
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

impl<'a> KernelLookup<MulScaled2Expr<'a>> for KernelLibrary {
	type CallType = MulScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulScaled2Expr<'a>>) -> MulScaledKernelCall<'a> {
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
		self.data.mul_scaled.call(a, b, scale1 * scale2)
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

	pub fn call<'a>(&'a self, x: &'a Tensor, y: &'a Tensor) -> MulXLnYKernelCall<'a> {
		MulXLnYKernelCall { kernel: self, x, y }
	}
}

pub struct MulXLnYKernelCall<'a> {
	kernel: &'a MulXLnYKernel,
	x: &'a Tensor,
	y: &'a Tensor,
}

impl<'a> EvaluatesToTensor for MulXLnYKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.x, self.y], [], [])
	}
}

type MulXLnYExpr<'a> =
	MulLookupExpr<
		&'a Tensor,
		LnClampedLookupExpr<
			&'a Tensor,
		>,
	>;

impl<'a> KernelLookup<MulXLnYExpr<'a>> for KernelLibrary {
	type CallType = MulXLnYKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulXLnYExpr<'a>>) -> MulXLnYKernelCall<'a> {
		let MulLookupExpr(
			x,
			LnClampedLookupExpr(
				y,
			),
		) = expr.0;
		self.data.mul_x_ln_y.call(x, y)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, a_weight: f64, b: &'a Tensor, b_weight: f64) -> WeightedAddKernelCall<'a> {
		WeightedAddKernelCall { kernel: self, a, a_weight, b, b_weight }
	}
}

pub struct WeightedAddKernelCall<'a> {
	kernel: &'a WeightedAddKernel,
	a: &'a Tensor,
	a_weight: f64,
	b: &'a Tensor,
	b_weight: f64,
}

impl<'a> EvaluatesToTensor for WeightedAddKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b], [], [self.a_weight, self.b_weight])
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

impl<'a> KernelLookup<WeightedAddExpr<'a>> for KernelLibrary {
	type CallType = WeightedAddKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<WeightedAddExpr<'a>>) -> WeightedAddKernelCall<'a> {
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
		self.data.weighted_add.call(a, a_weight, b, b_weight)
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

impl<'a> KernelLookup<WeightedSubExpr<'a>> for KernelLibrary {
	type CallType = WeightedAddKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<WeightedSubExpr<'a>>) -> WeightedAddKernelCall<'a> {
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
		self.data.weighted_add.call(a, a_weight, b, -b_weight)
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

	pub fn call<'a>(&'a self, x: &'a Tensor, a: &'a Tensor, b: &'a Tensor, scale: f64) -> AddXMulScaledKernelCall<'a> {
		AddXMulScaledKernelCall { kernel: self, x, a, b, scale }
	}
}

pub struct AddXMulScaledKernelCall<'a> {
	kernel: &'a AddXMulScaledKernel,
	x: &'a Tensor,
	a: &'a Tensor,
	b: &'a Tensor,
	scale: f64,
}

impl<'a> EvaluatesToTensor for AddXMulScaledKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.x, self.a, self.b], [], [self.scale])
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

impl<'a> KernelLookup<AddXMulScaledExpr<'a>> for KernelLibrary {
	type CallType = AddXMulScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<AddXMulScaledExpr<'a>>) -> AddXMulScaledKernelCall<'a> {
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
		self.data.add_x_mul_scaled.call(x, a, b, scale)
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

impl<'a> KernelLookup<AddXMulScaled2Expr<'a>> for KernelLibrary {
	type CallType = AddXMulScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<AddXMulScaled2Expr<'a>>) -> AddXMulScaledKernelCall<'a> {
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
		self.data.add_x_mul_scaled.call(x, a, b, scale1 * scale2)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor) -> DotKernelCall<'a> {
		DotKernelCall { kernel: self, a, b }
	}
}

pub struct DotKernelCall<'a> {
	kernel: &'a DotKernel,
	a: &'a Tensor,
	b: &'a Tensor,
}

impl<'a> EvaluatesToTensor for DotKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [], [self.a, self.b], [])
	}
}

type DotExpr<'a> =
	SumLookupExpr<
		MulLookupExpr<
			&'a Tensor,
			&'a Tensor,
		>,
	>;

impl<'a> KernelLookup<DotExpr<'a>> for KernelLibrary {
	type CallType = DotKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<DotExpr<'a>>) -> DotKernelCall<'a> {
		let SumLookupExpr(
			MulLookupExpr(
				a,
				b,
			),
		) = expr.0;
		self.data.dot.call(a, b)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor, scale: f64) -> DotScaledKernelCall<'a> {
		DotScaledKernelCall { kernel: self, a, b, scale }
	}
}

pub struct DotScaledKernelCall<'a> {
	kernel: &'a DotScaledKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	scale: f64,
}

impl<'a> EvaluatesToTensor for DotScaledKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [], [self.a, self.b], [self.scale])
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

impl<'a> KernelLookup<DotScaledExpr<'a>> for KernelLibrary {
	type CallType = DotScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<DotScaledExpr<'a>>) -> DotScaledKernelCall<'a> {
		let MulLookupExpr(
			SumLookupExpr(
				MulLookupExpr(
					a,
					b,
				),
			),
			scale,
		) = expr.0;
		self.data.dot_scaled.call(a, b, scale)
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

impl<'a> KernelLookup<DotScaled2Expr<'a>> for KernelLibrary {
	type CallType = DotScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<DotScaled2Expr<'a>>) -> DotScaledKernelCall<'a> {
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
		self.data.dot_scaled.call(a, b, scale1 * scale2)
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

	pub fn call<'a>(&'a self, t: &'a Tensor, t_weight: f64, a: &'a Tensor, b: &'a Tensor, ab_weight: f64) -> WeightedAddTDotKernelCall<'a> {
		WeightedAddTDotKernelCall { kernel: self, t, t_weight, a, b, ab_weight }
	}
}

pub struct WeightedAddTDotKernelCall<'a> {
	kernel: &'a WeightedAddTDotKernel,
	t: &'a Tensor,
	t_weight: f64,
	a: &'a Tensor,
	b: &'a Tensor,
	ab_weight: f64,
}

impl<'a> EvaluatesToTensor for WeightedAddTDotKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.t], [self.a, self.b], [self.t_weight, self.ab_weight])
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

impl<'a> KernelLookup<WeightedAddTDotExpr<'a>> for KernelLibrary {
	type CallType = WeightedAddTDotKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<WeightedAddTDotExpr<'a>>) -> WeightedAddTDotKernelCall<'a> {
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
		self.data.weighted_add_t_dot.call(t, t_weight, a, b, ab_weight)
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

impl<'a> KernelLookup<WeightedAddTDot2Expr<'a>> for KernelLibrary {
	type CallType = WeightedAddTDotKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<WeightedAddTDot2Expr<'a>>) -> WeightedAddTDotKernelCall<'a> {
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
		self.data.weighted_add_t_dot.call(t, t_weight, a, b, ab_weight1 * ab_weight2)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor, c: &'a Tensor, d: &'a Tensor) -> MulSubAMulBCDKernelCall<'a> {
		MulSubAMulBCDKernelCall { kernel: self, a, b, c, d }
	}
}

pub struct MulSubAMulBCDKernelCall<'a> {
	kernel: &'a MulSubAMulBCDKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	c: &'a Tensor,
	d: &'a Tensor,
}

impl<'a> EvaluatesToTensor for MulSubAMulBCDKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b, self.c, self.d], [], [])
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

impl<'a> KernelLookup<MulSubAMulBCDExpr<'a>> for KernelLibrary {
	type CallType = MulSubAMulBCDKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulSubAMulBCDExpr<'a>>) -> MulSubAMulBCDKernelCall<'a> {
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
		self.data.mul_sub_a_mul_b_c_d.call(a, b, c, d)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, b: &'a Tensor, c: &'a Tensor) -> MulSubABCKernelCall<'a> {
		MulSubABCKernelCall { kernel: self, a, b, c }
	}
}

pub struct MulSubABCKernelCall<'a> {
	kernel: &'a MulSubABCKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	c: &'a Tensor,
}

impl<'a> EvaluatesToTensor for MulSubABCKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b, self.c], [], [])
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

impl<'a> KernelLookup<MulSubABCExpr<'a>> for KernelLibrary {
	type CallType = MulSubABCKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulSubABCExpr<'a>>) -> MulSubABCKernelCall<'a> {
		let MulLookupExpr(
			SubLookupExpr(
				a,
				b,
			),
			c,
		) = expr.0;
		self.data.mul_sub_a_b_c.call(a, b, c)
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

	pub fn call<'a>(&'a self, a: &'a Tensor, eps: f64) -> SqrtRecipKernelCall<'a> {
		SqrtRecipKernelCall { kernel: self, a, eps }
	}
}

pub struct SqrtRecipKernelCall<'a> {
	kernel: &'a SqrtRecipKernel,
	a: &'a Tensor,
	eps: f64,
}

impl<'a> EvaluatesToTensor for SqrtRecipKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a], [], [self.eps])
	}
}

type SqrtRecipExpr<'a> =
	RecipLookupExpr<
		SqrtLookupExpr<
			&'a Tensor,
		>,
		f64,
	>;

impl<'a> KernelLookup<SqrtRecipExpr<'a>> for KernelLibrary {
	type CallType = SqrtRecipKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<SqrtRecipExpr<'a>>) -> SqrtRecipKernelCall<'a> {
		let RecipLookupExpr(
			SqrtLookupExpr(
				a,
			),
			eps,
		) = expr.0;
		self.data.sqrt_recip.call(a, eps)
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

	pub fn call<'a>(&'a self, x: &'a Tensor, a: &'a Tensor, b: &'a Tensor, scale: f64) -> AccMulScaledKernelCall<'a> {
		AccMulScaledKernelCall { kernel: self, x, a, b, scale }
	}
}

pub struct AccMulScaledKernelCall<'a> {
	kernel: &'a AccMulScaledKernel,
	x: &'a Tensor,
	a: &'a Tensor,
	b: &'a Tensor,
	scale: f64,
}

impl<'a> EvaluatesToTensor for AccMulScaledKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.x, self.a, self.b], [], [self.scale])
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

impl<'a> KernelLookup<AccMulScaledExpr<'a>> for KernelLibrary {
	type CallType = AccMulScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<AccMulScaledExpr<'a>>) -> AccMulScaledKernelCall<'a> {
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
		self.data.acc_mul_scaled.call(x, a, b, scale)
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

impl<'a> KernelLookup<AccNegMulScaledExpr<'a>> for KernelLibrary {
	type CallType = AccMulScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<AccNegMulScaledExpr<'a>>) -> AccMulScaledKernelCall<'a> {
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
		self.data.acc_mul_scaled.call(x, a, b, -scale)
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

	pub fn call<'a>(&'a self, lin: &'a Tensor, gate: &'a Tensor) -> SwigluKernelCall<'a> {
		SwigluKernelCall { kernel: self, lin, gate }
	}
}

pub struct SwigluKernelCall<'a> {
	kernel: &'a SwigluKernel,
	lin: &'a Tensor,
	gate: &'a Tensor,
}

impl<'a> EvaluatesToTensor for SwigluKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.lin, self.gate], [], [])
	}
}

type SwigluExpr<'a> =
	MulLookupExpr<
		&'a Tensor,
		SwishLookupExpr<
			&'a Tensor,
		>,
	>;

impl<'a> KernelLookup<SwigluExpr<'a>> for KernelLibrary {
	type CallType = SwigluKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<SwigluExpr<'a>>) -> SwigluKernelCall<'a> {
		let MulLookupExpr(
			lin,
			SwishLookupExpr(
				gate,
			),
		) = expr.0;
		self.data.swiglu.call(lin, gate)
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

	pub fn call<'a>(&'a self, v: f64) -> FillKernelCall<'a> {
		FillKernelCall { kernel: self, v }
	}
}

pub struct FillKernelCall<'a> {
	kernel: &'a FillKernel,
	v: f64,
}

impl<'a> EvaluatesToTensor for FillKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [], [], [self.v])
	}
}

type FillExpr<'a> =
	f64;

impl<'a> KernelLookup<FillExpr<'a>> for KernelLibrary {
	type CallType = FillKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<FillExpr<'a>>) -> FillKernelCall<'a> {
		let v = expr.0;
		self.data.fill.call(v)
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

	pub fn call<'a>(&'a self, v: &'a Tensor) -> CopyKernelCall<'a> {
		CopyKernelCall { kernel: self, v }
	}
}

pub struct CopyKernelCall<'a> {
	kernel: &'a CopyKernel,
	v: &'a Tensor,
}

impl<'a> EvaluatesToTensor for CopyKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.v], [], [])
	}
}

type CopyExpr<'a> =
	&'a Tensor;

impl<'a> KernelLookup<CopyExpr<'a>> for KernelLibrary {
	type CallType = CopyKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<CopyExpr<'a>>) -> CopyKernelCall<'a> {
		let v = expr.0;
		self.data.copy.call(v)
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
