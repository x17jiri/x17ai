//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use askama::Template;

pub struct TensorArgTemplate {
	pub dtype: String,
}

#[derive(Template)]
#[template(escape = "none", path = "elemwise.cu")]
pub struct ElemwiseTemplate {
	pub internal_dtype: String,
	pub out_dtype: String,
	pub tensor_args: Vec<TensorArgTemplate>,
	pub scalar_args_count: usize,
	pub expr: String,
}

#[derive(Template)]
#[template(escape = "none", path = "reduce.cu")]
pub struct ReduceTemplate {
	pub internal_dtype: String,
	pub out_dtype: String,
	pub tensor_args: Vec<TensorArgTemplate>,
	pub reduce_args_count: usize,
	pub scalar_args_count: usize,
	pub pre_reduce_expr: String,
	pub post_reduce_expr: String,
	pub post_reduce_common: String,
	pub warp_size: usize,
	pub identity: &'static str,
	pub loop_reduce: &'static str,
	pub pairwise_reduce: &'static str,
}
