// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

#![allow(non_snake_case)]
#![allow(incomplete_features)]
#![allow(internal_features)]
#![allow(non_upper_case_globals)]
#![feature(new_range_api)]
#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(stmt_expr_attributes)]
#![feature(let_chains)]
#![feature(arbitrary_self_types)]
#![feature(dispatch_from_dyn)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(trait_upcasting)]
#![feature(cursor_split)]
#![warn(clippy::cast_lossless)]

pub mod format;
pub mod nn;
pub mod tensor;
