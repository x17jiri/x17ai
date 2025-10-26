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

#[derive(Template)]
#[template(escape = "none", path = "elemwise.cu")]
pub struct Elemwise1DTemplate {
	pub internal_dtype: String,
	pub out_dtype: String,
	pub elem_args: Vec<ElemwiseArgTemplate>,
	pub scalar_args_count: usize,
	pub expr: String,
}
