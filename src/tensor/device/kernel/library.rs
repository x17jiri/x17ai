//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::OnceLock;

use super::generated_kernels::KernelLibraryData;
use super::lookup::{KernelLookup, LookupExpr, LookupWrapper};

//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub struct KernelLibrary {
	pub data: &'static KernelLibraryData,
}

impl KernelLibrary {
	pub fn instance() -> Self {
		static data_instance: OnceLock<KernelLibraryData> = OnceLock::new();
		let data = data_instance.get_or_init(|| KernelLibraryData::new());
		Self { data }
	}

	pub fn lookup<Expr>(&self, expr: LookupWrapper<Expr>) -> <Self as KernelLookup<Expr>>::CallType
	where
		Expr: LookupExpr,
		Self: KernelLookup<Expr>,
	{
		self.create_call(expr)
	}
}

//--------------------------------------------------------------------------------------------------
