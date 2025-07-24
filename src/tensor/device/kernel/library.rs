//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::OnceLock;

use crate::tensor::Tensor;
use crate::tensor::device::kernel::add::{AddKernel, AddKernelCall};
use crate::tensor::device::kernel::add_x_mul_scaled::{
	AddXMulScaledKernel, AddXMulScaledKernelCall,
};
use crate::tensor::device::kernel::dot::{DotKernel, DotKernelCall};
use crate::tensor::device::kernel::dot_scaled::{DotScaledKernel, DotScaledKernelCall};
use crate::tensor::device::kernel::lookup::{
	AddLookupExpr, KernelLookup, LnLookupExpr, LookupExpr, LookupWrapper, MulLookupExpr,
	RecipLookupExpr, SqrtLookupExpr, SubLookupExpr, SumLookupExpr,
};
use crate::tensor::device::kernel::mul::{MulKernel, MulKernelCall};
use crate::tensor::device::kernel::mul_scaled::{MulScaledKernel, MulScaledKernelCall};
use crate::tensor::device::kernel::mul_sub_a_b_c::{MulSubABCKernel, MulSubABCKernelCall};
use crate::tensor::device::kernel::mul_sub_a_mul_b_c_d::{
	MulSubAMulBCDKernel, MulSubAMulBCDKernelCall,
};
use crate::tensor::device::kernel::mul_x_log_y::{MulXLnYKernel, MulXLnYKernelCall};
use crate::tensor::device::kernel::rms::{RMSKernel, RMSKernelCall};
use crate::tensor::device::kernel::rms_recip::{RMSRecipKernel, RMSRecipKernelCall};
use crate::tensor::device::kernel::sqrt_recip::{SqrtRecipKernel, SqrtRecipKernelCall};
use crate::tensor::device::kernel::weighted_add::{WeightedAddKernel, WeightedAddKernelCall};
use crate::tensor::device::kernel::weighted_add_x_dot::{
	WeightedAddXDotKernel, WeightedAddXDotKernelCall,
};

//--------------------------------------------------------------------------------------------------

pub struct KernelLibraryData {
	rms: RMSKernel,
	rms_recip: RMSRecipKernel,
	add: AddKernel,
	weighted_add: WeightedAddKernel,
	weighted_add_x_dot: WeightedAddXDotKernel,
	add_x_mul_scaled: AddXMulScaledKernel,
	mul: MulKernel,
	mul_scaled: MulScaledKernel,
	mul_x_ln_y: MulXLnYKernel,
	dot: DotKernel,
	dot_scaled: DotScaledKernel,
	mul_sub_a_mul_b_c_d: MulSubAMulBCDKernel,
	mul_sub_a_b_c: MulSubABCKernel,
	sqrt_recip: SqrtRecipKernel,
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
			weighted_add: WeightedAddKernel::instance(),
			weighted_add_x_dot: WeightedAddXDotKernel::instance(),
			add_x_mul_scaled: AddXMulScaledKernel::instance(),
			mul_scaled: MulScaledKernel::instance(),
			mul_x_ln_y: MulXLnYKernel::instance(),
			dot: DotKernel::instance(),
			dot_scaled: DotScaledKernel::instance(),
			mul_sub_a_mul_b_c_d: MulSubAMulBCDKernel::instance(),
			mul_sub_a_b_c: MulSubABCKernel::instance(),
			sqrt_recip: SqrtRecipKernel::instance(),
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

/// `a * b * c`
#[rustfmt::skip]
type MulScaledExpr<'a> =
	MulLookupExpr<
		MulLookupExpr<
			&'a Tensor,
			&'a Tensor
		>,
		f64
	>;

/// `a * b * c`
impl<'a> KernelLookup<MulScaledExpr<'a>> for KernelLibrary {
	type CallType = MulScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulScaledExpr<'a>>) -> MulScaledKernelCall<'a> {
		let MulLookupExpr(MulLookupExpr(a, b), c) = expr.0;
		self.data.mul_scaled.call(a, b, c)
	}
}

//--------------------------------------------------------------------------------------------------

/// `x * ln(y)`
#[rustfmt::skip]
type MulXLnYExpr<'a> =
	MulLookupExpr<
		&'a Tensor,
		LnLookupExpr<&'a Tensor>
	>;

/// `x * log(y)`
impl<'a> KernelLookup<MulXLnYExpr<'a>> for KernelLibrary {
	type CallType = MulXLnYKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<MulXLnYExpr<'a>>) -> MulXLnYKernelCall<'a> {
		let MulLookupExpr(x, LnLookupExpr(y)) = expr.0;
		self.data.mul_x_ln_y.call(x, y)
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

/// `x + (a * b * c)`
#[rustfmt::skip]
type AddXMulScaledExpr<'a> =
	AddLookupExpr<
		&'a Tensor,
		MulLookupExpr<
			MulLookupExpr<
				&'a Tensor,
				&'a Tensor
			>,
			f64
		>
	>;

/// `x + (a * b * c)`
impl<'a> KernelLookup<AddXMulScaledExpr<'a>> for KernelLibrary {
	type CallType = AddXMulScaledKernelCall<'a>;

	fn create_call(
		&self,
		expr: LookupWrapper<AddXMulScaledExpr<'a>>,
	) -> AddXMulScaledKernelCall<'a> {
		let AddLookupExpr(x, MulLookupExpr(MulLookupExpr(a, b), c)) = expr.0;
		self.data.add_x_mul_scaled.call(x, a, b, c)
	}
}

//--------------------------------------------------------------------------------------------------

/// `x + (a * b * c)`
#[rustfmt::skip]
type SubXMulScaledExpr<'a> =
	SubLookupExpr<
		&'a Tensor,
		MulLookupExpr<
			MulLookupExpr<
				&'a Tensor,
				&'a Tensor
			>,
			f64
		>
	>;

/// `x + (a * b * c)`
impl<'a> KernelLookup<SubXMulScaledExpr<'a>> for KernelLibrary {
	type CallType = AddXMulScaledKernelCall<'a>;

	fn create_call(
		&self,
		expr: LookupWrapper<SubXMulScaledExpr<'a>>,
	) -> AddXMulScaledKernelCall<'a> {
		let SubLookupExpr(x, MulLookupExpr(MulLookupExpr(a, b), c)) = expr.0;
		self.data.add_x_mul_scaled.call(x, a, b, -c)
	}
}

//--------------------------------------------------------------------------------------------------

/// `(a * b).sum()`
#[rustfmt::skip]
type DotExpr<'a> =
	SumLookupExpr<
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
type DotScaledExpr<'a> =
	MulLookupExpr<
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

/// `(a * b).sum() * c * d`
#[rustfmt::skip]
type DotScaled2Expr<'a> =
	MulLookupExpr<
		MulLookupExpr<
			SumLookupExpr<
				MulLookupExpr<
					&'a Tensor,
					&'a Tensor
				>
			>,
			f64
		>,
		f64
	>;

/// `(a * b).sum() * c * d`
impl<'a> KernelLookup<DotScaled2Expr<'a>> for KernelLibrary {
	type CallType = DotScaledKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<DotScaled2Expr<'a>>) -> DotScaledKernelCall<'a> {
		let MulLookupExpr(MulLookupExpr(SumLookupExpr(MulLookupExpr(a, b)), c), d) = expr.0;
		self.data.dot_scaled.call(a, b, c * d)
	}
}

//--------------------------------------------------------------------------------------------------

/// `(a * a_weight) + (b * b_weight)`
#[rustfmt::skip]
type WeightedAddExpr<'a> =
	AddLookupExpr<
		MulLookupExpr<&'a Tensor, f64>,
		MulLookupExpr<&'a Tensor, f64>
	>;

/// `(a * a_weight) + (b * b_weight)`
impl<'a> KernelLookup<WeightedAddExpr<'a>> for KernelLibrary {
	type CallType = WeightedAddKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<WeightedAddExpr<'a>>) -> WeightedAddKernelCall<'a> {
		let AddLookupExpr(MulLookupExpr(a, a_weight), MulLookupExpr(b, b_weight)) = expr.0;
		self.data.weighted_add.call(a, a_weight, b, b_weight)
	}
}

//--------------------------------------------------------------------------------------------------

/// `(a * a_weight) - (b * b_weight)`
#[rustfmt::skip]
type WeightedSubExpr<'a> =
	SubLookupExpr<
		MulLookupExpr<&'a Tensor, f64>,
		MulLookupExpr<&'a Tensor, f64>
	>;

/// `(a * a_weight) + (b * b_weight)`
impl<'a> KernelLookup<WeightedSubExpr<'a>> for KernelLibrary {
	type CallType = WeightedAddKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<WeightedSubExpr<'a>>) -> WeightedAddKernelCall<'a> {
		let SubLookupExpr(MulLookupExpr(a, a_weight), MulLookupExpr(b, b_weight)) = expr.0;
		self.data.weighted_add.call(a, a_weight, b, -b_weight)
	}
}

//--------------------------------------------------------------------------------------------------

/// `(a * a_weight) + ((b * c).sum() * dot_weight)`
#[rustfmt::skip]
type WeightedAddXDotExpr<'a> =
	AddLookupExpr<
		MulLookupExpr<&'a Tensor, f64>,
		MulLookupExpr<
			SumLookupExpr<MulLookupExpr<&'a Tensor, &'a Tensor>>,
			f64
		>
	>;

/// `(a * a_weight) + ((b * c).sum() * dot_weight)`
impl<'a> KernelLookup<WeightedAddXDotExpr<'a>> for KernelLibrary {
	type CallType = WeightedAddXDotKernelCall<'a>;

	fn create_call(
		&self,
		expr: LookupWrapper<WeightedAddXDotExpr<'a>>,
	) -> WeightedAddXDotKernelCall<'a> {
		let AddLookupExpr(
			MulLookupExpr(a, a_weight),
			MulLookupExpr(SumLookupExpr(MulLookupExpr(b, c)), dot_weight),
		) = expr.0;
		self.data.weighted_add_x_dot.call(a, a_weight, b, c, dot_weight)
	}
}

//--------------------------------------------------------------------------------------------------

/// `(a * a_weight) + ((b * c).sum() * dot_weight1 * dot_weight2)`
#[rustfmt::skip]
type WeightedAddXDotScaledExpr<'a> =
	AddLookupExpr<
		MulLookupExpr<&'a Tensor, f64>,
		MulLookupExpr<
			MulLookupExpr<
				SumLookupExpr<MulLookupExpr<&'a Tensor, &'a Tensor>>,
				f64
			>,
			f64
		>
	>;

/// `(a * a_weight) + ((b * c).sum() * dot_weight1 * dot_weight2)`
impl<'a> KernelLookup<WeightedAddXDotScaledExpr<'a>> for KernelLibrary {
	type CallType = WeightedAddXDotKernelCall<'a>;

	fn create_call(
		&self,
		expr: LookupWrapper<WeightedAddXDotScaledExpr<'a>>,
	) -> WeightedAddXDotKernelCall<'a> {
		#[rustfmt::skip]
		let AddLookupExpr(
			MulLookupExpr(a, a_weight),
			MulLookupExpr(
				MulLookupExpr(
					SumLookupExpr(MulLookupExpr(b, c)),
					dot_weight1
				),
				dot_weight2
			),
		) = expr.0;
		self.data.weighted_add_x_dot.call(a, a_weight, b, c, dot_weight1 * dot_weight2)
	}
}

//--------------------------------------------------------------------------------------------------

/// `inp.sqrt().recip(eps)`
#[rustfmt::skip]
type SqrtRecipExpr<'a> =
	RecipLookupExpr<
		SqrtLookupExpr<&'a Tensor>,
		f64
	>;

/// `inp.sqrt().recip(eps)`
impl<'a> KernelLookup<SqrtRecipExpr<'a>> for KernelLibrary {
	type CallType = SqrtRecipKernelCall<'a>;

	fn create_call(&self, expr: LookupWrapper<SqrtRecipExpr<'a>>) -> SqrtRecipKernelCall<'a> {
		let RecipLookupExpr(SqrtLookupExpr(inp), eps) = expr.0;
		self.data.sqrt_recip.call(inp, eps)
	}
}

//--------------------------------------------------------------------------------------------------
