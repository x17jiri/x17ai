//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

// TODO: find all uses of `#[allow(clippy::indexing_slicing)]`
// and make sure Rust doesn't generate any panics

#![allow(non_snake_case)]
#![allow(incomplete_features)]
#![allow(internal_features)]
#![allow(non_upper_case_globals)]
#![feature(vec_push_within_capacity)]
#![feature(try_blocks)]
#![feature(allocator_api)]
#![feature(f16)]
#![feature(slice_ptr_get)]
#![feature(new_range_api)]
#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(stmt_expr_attributes)]
#![feature(arbitrary_self_types)]
#![feature(dispatch_from_dyn)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(cursor_split)]
#![feature(negative_impls)]
#![feature(cold_path)]
#![feature(likely_unlikely)]
#![feature(auto_traits)]
#![feature(coerce_unsized)]
#![feature(unsize)]
#![feature(array_try_map)]
#![feature(array_try_from_fn)]
#![feature(maybe_uninit_uninit_array_transpose)]
#![feature(associated_type_defaults)]
#![feature(box_into_inner)]
#![feature(unboxed_closures)]
#![feature(specialization)]
#![feature(const_trait_impl)]
#![feature(ptr_as_ref_unchecked)]
#![feature(const_option_ops)]
#![feature(get_mut_unchecked)]
#![feature(ptr_metadata)]
#![feature(macro_metavar_expr)]
#![feature(trait_alias)]
#![feature(thin_box)]
#![feature(const_index)]
#![feature(string_from_utf8_lossy_owned)]
#![feature(int_roundings)]
// clippy
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::cargo)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::panic_in_result_fn)]
#![warn(clippy::panic)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::comparison_chain)]
#![allow(non_camel_case_types)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::large_digit_groups)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::let_and_return)]
#![allow(clippy::inline_always)]
#![allow(clippy::needless_lifetimes)]
#![allow(unused_parens)]
#![allow(clippy::tabs_in_doc_comments)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::range_plus_one)]
//#![allow(clippy::len_zero)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(irrefutable_let_patterns)]
#![allow(clippy::useless_format)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::useless_let_if_seq)]
use std::borrow::Cow;
use std::convert::Infallible;

pub mod autograd;
pub mod new;
pub mod nn;
pub mod rng;
pub mod tensor;
pub mod util;

#[derive(Debug)]
pub struct ErrExtra {
	pub message: Cow<'static, str>,
	pub nested: Option<Box<dyn std::error::Error + Send + Sync>>,
}

#[derive(Debug)]
pub struct ErrPack<Code: Copy + std::fmt::Debug> {
	pub code: Code,
	pub extra: Option<Box<ErrExtra>>,
}

#[cold]
#[inline(never)]
#[allow(clippy::panic)]
fn panic_infallible_to_err_conversion<Code: Copy + std::fmt::Debug>() -> ErrPack<Code> {
	panic!("Infallible should never be converted to ErrPack");
}

impl<Code: Copy + std::fmt::Debug> From<Infallible> for ErrPack<Code> {
	fn from(_: Infallible) -> Self {
		panic_infallible_to_err_conversion()
	}
}

impl<Code: Copy + std::fmt::Debug> std::error::Error for ErrPack<Code> {
}

impl<Code: Copy + std::fmt::Debug> std::fmt::Display for ErrPack<Code> {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		let code = self.code;
		write!(f, "(ErrPack: code={code:?}")?;
		if let Some(ref extra) = self.extra {
			let msg = extra.message.as_ref();
			if !msg.is_empty() {
				write!(f, ", message={msg}")?;
			}
			if let Some(nested) = &extra.nested {
				write!(f, ", nested={nested:?}")?;
			}
		}
		write!(f, ")")
	}
}
