//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::num::NonZeroUsize;

use askama::Template;

//--------------------------------------------------------------------------------------------------

pub struct BasicGemmWriterTemplate {
	pub use_l2_norm: bool,
	pub use_geglu: bool,
	pub c_type: &'static str,
	pub c_stride_expr: &'static str,
	pub scale_val: String,
	pub scale_dscr: String,
	pub head_dim: usize,
	pub sep_dim: usize,
	pub eps_val: String,
	pub head_scale_val: String,
	pub head_scale_dscr: String,
	pub sep_scale_val: String,
	pub sep_scale_dscr: String,
	pub geglu_inp_scale_val: String,
	pub geglu_inp_scale_dscr: String,
	pub geglu_out_scale_val: String,
	pub geglu_out_scale_dscr: String,
	pub has_rrms_output: bool,
}

#[derive(Template)]
#[template(escape = "none", path = "basic_gemm/common.cuh")]
pub struct BasicGemmCommonTemplate<'a> {
	pub a_cols: usize,
	pub b_rows: Option<NonZeroUsize>,
	pub writer: &'a BasicGemmWriterTemplate,
}

#[derive(Template)]
#[template(escape = "none", path = "basic_gemm/kernel.cu")]
pub struct BasicGemmKernelTemplate<'a> {
	pub a_cols: usize,
	pub b_rows: Option<NonZeroUsize>,
	pub writer: &'a BasicGemmWriterTemplate,
}

#[derive(Template)]
#[template(escape = "none", path = "basic_gemm/meta.cu")]
pub struct BasicGemmMetaTemplate {}

//--------------------------------------------------------------------------------------------------
