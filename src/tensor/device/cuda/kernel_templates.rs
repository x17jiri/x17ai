//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use askama::Template;

pub struct ElemwiseArgTemplate {
	pub dtype: String,
}

pub struct ReduceArgTemplate {
	pub dtype: String,
}

#[derive(Template)]
#[template(escape = "none", path = "elemwise.cu")]
pub struct ElemwiseTemplate {
	pub internal_dtype: String,
	pub out_dtype: String,
	pub elem_args: Vec<ElemwiseArgTemplate>,
	pub scalar_args_count: usize,
	pub expr: String,
}

#[derive(Template)]
#[template(escape = "none", path = "reduce.cu")]
pub struct ReduceTemplate {
	pub internal_dtype: String,
	pub out_dtype: String,
	pub reduce_args: Vec<ReduceArgTemplate>,
	pub elem_args: Vec<ElemwiseArgTemplate>,
	pub scalar_args_count: usize,
	pub pre_reduce_expr: String,
	pub post_reduce_expr: String,
	pub zero: String,
	pub warp_size: usize,
}
