//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::OnceLock;

use crate::tensor::Tensor;
use crate::tensor::device::kernel::add::{AddKernel, AddKernelCall};
use crate::tensor::device::kernel::dot::{DotKernel, DotKernelCall};
use crate::tensor::device::kernel::dot_scaled::{DotScaledKernel, DotScaledKernelCall};
use crate::tensor::device::kernel::lookup::{
	AddLookupExpr, KernelLookup, LookupExpr, LookupWrapper, MulLookupExpr, SubLookupExpr,
	SumLookupExpr,
};
use crate::tensor::device::kernel::mul::{MulKernel, MulKernelCall};
use crate::tensor::device::kernel::mul_scaled::{MulScaledKernel, MulScaledKernelCall};
use crate::tensor::device::kernel::mul_sub_a_b_c::{MulSubABCKernel, MulSubABCKernelCall};
use crate::tensor::device::kernel::mul_sub_a_mul_b_c_d::{
	MulSubAMulBCDKernel, MulSubAMulBCDKernelCall,
};
use crate::tensor::device::kernel::rms::{RMSKernel, RMSKernelCall};
use crate::tensor::device::kernel::rms_recip::{RMSRecipKernel, RMSRecipKernelCall};

//--------------------------------------------------------------------------------------------------

pub struct KernelLibraryData {
	rms: RMSKernel,
	rms_recip: RMSRecipKernel,
	add: AddKernel,
	mul: MulKernel,
	mul_scaled: MulScaledKernel,
	dot: DotKernel,
	dot_scaled: DotScaledKernel,
	mul_sub_a_mul_b_c_d: MulSubAMulBCDKernel,
	mul_sub_a_b_c: MulSubABCKernel,
}

#[derive(Copy, Clone)]
pub struct KernelLibrary {
	data: &'static KernelLibraryData,
}

impl KernelLibrary {
	pub fn instance() -> Self {
		static data_instance: OnceLock<KernelLibraryData> = OnceLock::new();
		let data = data_instance.get_or_init(|| KernelLibraryData {
			rms: RMSKernel::instance(),
			mul: MulKernel::instance(),
			rms_recip: RMSRecipKernel::instance(),
			add: AddKernel::instance(),
			mul_scaled: MulScaledKernel::instance(),
			dot: DotKernel::instance(),
			dot_scaled: DotScaledKernel::instance(),
			mul_sub_a_mul_b_c_d: MulSubAMulBCDKernel::instance(),
			mul_sub_a_b_c: MulSubABCKernel::instance(),
		});
		Self { data }
	}

	pub fn lookup<Expr>(&self, expr: LookupWrapper<Expr>) -> <Self as KernelLookup<Expr>>::CallType
	where
		Expr: LookupExpr,
		Self: KernelLookup<Expr>,
	{
		self.create_call(expr)
	}

	/// # Expression
	///
	///     (inp * inp).mean().sqrt()
	pub fn rms<'a>(&self, inp: &'a Tensor) -> RMSKernelCall<'a> {
		self.data.rms.call(inp)
	}

	/// # Expression
	///
	///     1.0 / ((inp * inp).mean().sqrt() + eps)
	pub fn rms_recip<'a>(&self, inp: &'a Tensor, eps: f64) -> RMSRecipKernelCall<'a> {
		self.data.rms_recip.call(inp, eps)
	}

	/// # Expression
	///
	///     a * b
	pub fn mul<'a>(&self, a: &'a Tensor, b: &'a Tensor) -> MulKernelCall<'a> {
		self.data.mul.call(a, b)
	}

	/// # Expression
	///
	///     a * b * scale
	pub fn mul_scaled<'a>(
		&self,
		a: &'a Tensor,
		b: &'a Tensor,
		scale: f64,
	) -> MulScaledKernelCall<'a> {
		self.data.mul_scaled.call(a, b, scale)
	}

	/// # Expression
	///
	///     (a * b).sum()
	pub fn dot<'a>(&self, a: &'a Tensor, b: &'a Tensor) -> DotKernelCall<'a> {
		self.data.dot.call(a, b)
	}

	/// # Expression
	///
	///     (a * b).sum() * scale
	pub fn dot_scaled<'a>(
		&self,
		a: &'a Tensor,
		b: &'a Tensor,
		scale: f64,
	) -> DotScaledKernelCall<'a> {
		self.data.dot_scaled.call(a, b, scale)
	}

	/// # Expression
	///
	///     (a - (b * c)) * d
	pub fn mul_sub_a_mul_b_c_d<'a>(
		&self,
		a: &'a Tensor,
		b: &'a Tensor,
		c: &'a Tensor,
		d: &'a Tensor,
	) -> MulSubAMulBCDKernelCall<'a> {
		self.data.mul_sub_a_mul_b_c_d.call(a, b, c, d)
	}

	/// # Expression
	///
	///      (a - b) * c
	pub fn mul_sub_a_b_c<'a>(
		&self,
		a: &'a Tensor,
		b: &'a Tensor,
		c: &'a Tensor,
	) -> MulSubABCKernelCall<'a> {
		self.data.mul_sub_a_b_c.call(a, b, c)
	}
}

//--------------------------------------------------------------------------------------------------

/// `(a - (b * c)) * d`
#[rustfmt::skip]
type MulSubAMulBCDExpr<'a> =
	MulLookupExpr<
		SubLookupExpr<
			&'a Tensor,
			MulLookupExpr<
				&'a Tensor,
				&'a Tensor
			>
		>,
		&'a Tensor
	>;

/// `(a - (b * c)) * d`
impl<'a> KernelLookup<MulSubAMulBCDExpr<'a>> for KernelLibrary {
	type CallType = MulSubAMulBCDKernelCall<'a>;

	fn create_call(
		&self,
		expr: LookupWrapper<MulSubAMulBCDExpr<'a>>,
	) -> MulSubAMulBCDKernelCall<'a> {
		let MulLookupExpr(SubLookupExpr(a, MulLookupExpr(b, c)), d) = expr.0;
		self.data.mul_sub_a_mul_b_c_d.call(a, b, c, d)
	}
}

//--------------------------------------------------------------------------------------------------

/// `(a - b) * c`
#[rustfmt::skip]
type MulSubABCExpr<'a> =
	MulLookupExpr<
		SubLookupExpr<
			&'a Tensor,
			&'a Tensor,
		>,
		&'a Tensor
	>;

/// `(a - b) * c`
impl<'a> KernelLookup<MulSubABCExpr<'a>> for KernelLibrary {
	type CallType = MulSubABCKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulSubABCExpr<'a>>) -> MulSubABCKernelCall<'a> {
		let MulLookupExpr(SubLookupExpr(a, b), c) = expr.0;
		self.data.mul_sub_a_b_c.call(a, b, c)
	}
}

//--------------------------------------------------------------------------------------------------

/// `a * b`
type MulExpr<'a> = MulLookupExpr<&'a Tensor, &'a Tensor>;

/// `a * b`
impl<'a> KernelLookup<MulExpr<'a>> for KernelLibrary {
	type CallType = MulKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulExpr<'a>>) -> MulKernelCall<'a> {
		let MulLookupExpr(a, b) = expr.0;
		self.data.mul.call(a, b)
	}
}

//--------------------------------------------------------------------------------------------------

/// `a + b`
type AddExpr<'a> = AddLookupExpr<&'a Tensor, &'a Tensor>;

/// `a + b`
impl<'a> KernelLookup<AddExpr<'a>> for KernelLibrary {
	type CallType = AddKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<AddExpr<'a>>) -> AddKernelCall<'a> {
		let AddLookupExpr(a, b) = expr.0;
		self.data.add.call(a, b)
	}
}

//--------------------------------------------------------------------------------------------------

/// `(a * b).sum()`
#[rustfmt::skip]
type DotExpr<'a> = SumLookupExpr<
	MulLookupExpr<
		&'a Tensor,
		&'a Tensor
	>
>;

/// `(a * b).sum()`
impl<'a> KernelLookup<DotExpr<'a>> for KernelLibrary {
	type CallType = DotKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<DotExpr<'a>>) -> DotKernelCall<'a> {
		let SumLookupExpr(MulLookupExpr(a, b)) = expr.0;
		self.data.dot.call(a, b)
	}
}

//--------------------------------------------------------------------------------------------------

/// `(a * b).sum() * c`
#[rustfmt::skip]
type DotScaledExpr<'a> = MulLookupExpr<
	SumLookupExpr<
		MulLookupExpr<
			&'a Tensor,
			&'a Tensor
		>
	>,
	f64
>;

/// `(a * b).sum() * c`
impl<'a> KernelLookup<DotScaledExpr<'a>> for KernelLibrary {
	type CallType = DotScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<DotScaledExpr<'a>>) -> DotScaledKernelCall<'a> {
		let MulLookupExpr(SumLookupExpr(MulLookupExpr(a, b)), c) = expr.0;
		self.data.dot_scaled.call(a, b, c)
	}
}

//--------------------------------------------------------------------------------------------------
