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
use super::lookup::{AddLookupExpr, KernelLookup, LookupWrapper, MulLookupExpr, SumLookupExpr};

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

	pub fn call<'a>(&'a self, a_weight: f64, a: &'a Tensor, b_weight: f64, b: &'a Tensor) -> WeightedAddKernelCall<'a> {
		WeightedAddKernelCall { kernel: self, a_weight, a, b_weight, b }
	}
}

pub struct WeightedAddKernelCall<'a> {
	kernel: &'a WeightedAddKernel,
	a_weight: f64,
	a: &'a Tensor,
	b_weight: f64,
	b: &'a Tensor,
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
		self.data.weighted_add.call(a_weight, a, b_weight, b)
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

pub struct KernelLibraryData {
	add: AddKernel,
	weighted_add: WeightedAddKernel,
	dot: DotKernel,
	dot_scaled: DotScaledKernel,
}

impl KernelLibraryData {
	pub fn new() -> Self {
		Self {
			add: AddKernel::new(),
			weighted_add: WeightedAddKernel::new(),
			dot: DotKernel::new(),
			dot_scaled: DotScaledKernel::new(),
		}
	}
}

//--------------------------------------------------------------------------------------------------
