//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::OnceLock;

use crate::tensor::Tensor;
use crate::tensor::device::kernel::Ia_sub_Ib_mul_cII_mul_d::{
	Ia_sub_Ib_mul_cII_mul_d_Kernel, Ia_sub_Ib_mul_cII_mul_d_KernelCall,
};
use crate::tensor::device::kernel::dot::{DotKernel, DotKernelCall};
use crate::tensor::device::kernel::dot_scaled::{DotScaledKernel, DotScaledKernelCall};
use crate::tensor::device::kernel::lookup::{
	self, KernelLookup, LookupExpr, MulLookupExpr, SubLookupExpr,
};
use crate::tensor::device::kernel::mul::{MulKernel, MulKernelCall};
use crate::tensor::device::kernel::mul_scaled::{MulScaledKernel, MulScaledKernelCall};
use crate::tensor::device::kernel::rms::{RMSKernel, RMSKernelCall};
use crate::tensor::device::kernel::rms_recip::{RMSRecipKernel, RMSRecipKernelCall};

//--------------------------------------------------------------------------------------------------

pub struct KernelLibraryData {
	rms: RMSKernel,
	rms_recip: RMSRecipKernel,
	mul: MulKernel,
	mul_scaled: MulScaledKernel,
	dot: DotKernel,
	dot_scaled: DotScaledKernel,
	Ia_sub_Ib_mul_cII_mul_d: Ia_sub_Ib_mul_cII_mul_d_Kernel,
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
			rms_recip: RMSRecipKernel::instance(),
			mul: MulKernel::instance(),
			mul_scaled: MulScaledKernel::instance(),
			dot: DotKernel::instance(),
			dot_scaled: DotScaledKernel::instance(),
			Ia_sub_Ib_mul_cII_mul_d: Ia_sub_Ib_mul_cII_mul_d_Kernel::instance(),
		});
		Self { data }
	}

	pub fn lookup<Expr>(
		&self,
		expr: lookup::Wrapper<Expr>,
	) -> <Self as KernelLookup<Expr>>::CallType
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
	pub fn Ia_sub_Ib_mul_cII_mul_d<'a>(
		&self,
		a: &'a Tensor,
		b: &'a Tensor,
		c: &'a Tensor,
		d: &'a Tensor,
	) -> Ia_sub_Ib_mul_cII_mul_d_KernelCall<'a> {
		self.data.Ia_sub_Ib_mul_cII_mul_d.call(a, b, c, d)
	}
}

//--------------------------------------------------------------------------------------------------

#[rustfmt::skip]
type Ia_sub_Ib_mul_cII_mul_d_Expr<'a> =
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

/// (a - (b * c)) * d
impl<'a> KernelLookup<Ia_sub_Ib_mul_cII_mul_d_Expr<'a>> for KernelLibrary {
	type CallType = Ia_sub_Ib_mul_cII_mul_d_KernelCall<'a>;

	fn create_call(
		&self,
		expr: lookup::Wrapper<Ia_sub_Ib_mul_cII_mul_d_Expr<'a>>,
	) -> Ia_sub_Ib_mul_cII_mul_d_KernelCall<'a> {
		let MulLookupExpr(SubLookupExpr(a, MulLookupExpr(b, c)), d) = expr.0;
		self.data.Ia_sub_Ib_mul_cII_mul_d.call(a, b, c, d)
	}
}

//--------------------------------------------------------------------------------------------------
