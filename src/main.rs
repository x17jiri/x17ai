// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]

#[cold]
fn cold_path() {}

pub enum Error {
	TooManyDims,
	TooManyElems,
}

mod cpu_dev;
mod shape;
mod tensor;
mod x17ai;
