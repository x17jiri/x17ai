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

mod buffer;
mod cpu_dev;
mod device;
mod shape;
mod x17ai;
mod tensor;
