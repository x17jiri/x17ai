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
#![feature(map_try_insert)]
#![feature(str_as_str)]
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
#![allow(clippy::len_zero)]
#![allow(clippy::ref_option)]
#![allow(clippy::result_unit_err)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::needless_else)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::single_match_else)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_let_else)]

use std::alloc::AllocError;
use std::borrow::Cow;
use std::convert::Infallible;

pub mod device;
pub mod dtype;
pub mod literal;
pub mod shape;
pub mod tensor;
pub mod util;

#[derive(Copy, Clone, Debug)]
pub struct ShapeOverflowError;

#[derive(Copy, Clone, Debug)]
pub struct DeviceAllocError;

#[derive(Copy, Clone, Debug)]
pub struct KernelGeneratorError;

#[derive(Copy, Clone, Debug)]
pub enum TensorOpError {
	ShapeOverflow,
	Alloc,
	DeviceAlloc,
	UnsupportedDType,
	IOError,
	InvalidSafeTensors,
	Device,
	KernelGenerator,
	Other,
}

impl TensorOpError {
	pub fn new_io_error(message: String) -> ErrPack<Self> {
		ErrPack::new(Self::IOError, message)
	}
}

impl From<ShapeOverflowError> for TensorOpError {
	fn from(_: ShapeOverflowError) -> TensorOpError {
		TensorOpError::ShapeOverflow
	}
}

impl From<ShapeOverflowError> for ErrPack<TensorOpError> {
	fn from(_: ShapeOverflowError) -> ErrPack<TensorOpError> {
		ErrPack {
			code: TensorOpError::ShapeOverflow,
			extra: None,
		}
	}
}

impl From<DeviceAllocError> for TensorOpError {
	fn from(_: DeviceAllocError) -> TensorOpError {
		TensorOpError::DeviceAlloc
	}
}

impl From<DeviceAllocError> for ErrPack<TensorOpError> {
	fn from(_: DeviceAllocError) -> ErrPack<TensorOpError> {
		ErrPack {
			code: TensorOpError::DeviceAlloc,
			extra: None,
		}
	}
}

impl From<dtype::UnsupportedDTypeError> for TensorOpError {
	fn from(_: dtype::UnsupportedDTypeError) -> TensorOpError {
		TensorOpError::UnsupportedDType
	}
}

impl From<dtype::UnsupportedDTypeError> for ErrPack<TensorOpError> {
	fn from(_: dtype::UnsupportedDTypeError) -> ErrPack<TensorOpError> {
		ErrPack {
			code: TensorOpError::UnsupportedDType,
			extra: None,
		}
	}
}

impl From<std::io::Error> for ErrPack<TensorOpError> {
	fn from(err: std::io::Error) -> ErrPack<TensorOpError> {
		ErrPack::new_with_nested(TensorOpError::IOError, "failed to read tensor file", err)
	}
}

impl From<safetensors::SafeTensorError> for ErrPack<TensorOpError> {
	fn from(err: safetensors::SafeTensorError) -> ErrPack<TensorOpError> {
		ErrPack::new_with_nested(TensorOpError::InvalidSafeTensors, "invalid safetensors file", err)
	}
}

impl From<AllocError> for TensorOpError {
	fn from(_: AllocError) -> TensorOpError {
		TensorOpError::Alloc
	}
}

impl From<AllocError> for ErrPack<TensorOpError> {
	fn from(_: AllocError) -> ErrPack<TensorOpError> {
		ErrPack { code: TensorOpError::Alloc, extra: None }
	}
}

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

impl<Code: Copy + std::fmt::Debug> ErrPack<Code> {
	pub fn new(code: Code, message: impl Into<Cow<'static, str>>) -> Self {
		Self {
			code,
			extra: Some(Box::new(ErrExtra {
				message: message.into(),
				nested: None,
			})),
		}
	}

	pub fn new_with_nested(
		code: Code,
		message: impl Into<Cow<'static, str>>,
		nested: impl std::error::Error + Send + Sync + 'static,
	) -> Self {
		Self {
			code,
			extra: Some(Box::new(ErrExtra {
				message: message.into(),
				nested: Some(Box::new(nested)),
			})),
		}
	}
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

pub struct Diagnostic {
	pub is_error: bool,
	pub message: String,
}

pub struct Diagnostics {
	pub list: Vec<Diagnostic>,
	pub err_count: usize,
}

impl Diagnostics {
	pub fn new() -> Self {
		Self { list: Vec::new(), err_count: 0 }
	}

	pub fn add_error(&mut self, message: String) {
		self.err_count += 1;
		self.list.push(Diagnostic { is_error: true, message });
	}
}

impl Default for Diagnostics {
	fn default() -> Self {
		Self::new()
	}
}
