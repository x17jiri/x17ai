//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::intrinsics::cold_path;

use log::warn;

use crate::{Error, Result};

use super::dim_merger::{MergedDim, MergedDimIter};

#[inline(never)]
fn __run_recursive<const N: usize>(
	prev_dim: MergedDim<N>, mut dims: impl Clone + Iterator<Item = MergedDim<N>>,
	offsets: [usize; N],
	f: &mut impl FnMut(
		usize,      // batch_size
		[usize; N], // batch_strides
		[usize; N], // offsets
	) -> Result<()>,
) -> Result<()> {
	if prev_dim.size > 1 && prev_dim.strides[0] == 0 {
		#[cold]
		fn err_cannot_broadcast_output_tensor() -> Error {
			"broadcast is disabled for output tensor".into()
		}
		return Err(err_cannot_broadcast_output_tensor());
	}

	if let Some(dim) = dims.next() {
		let mut offsets = offsets;
		for _ in 0..prev_dim.size {
			__run_recursive(dim, dims.clone(), offsets, f)?;

			for j in 0..N {
				offsets[j] += prev_dim.strides[j];
			}
		}
		Ok(())
	} else {
		f(prev_dim.size, prev_dim.strides, offsets)
	}
}

#[inline(never)]
fn __run<const N: usize>(
	mut dims: MergedDimIter<N>, offsets: [usize; N],
	mut f: impl FnMut(
		usize,      // batch_size
		[usize; N], // batch_strides
		[usize; N], // offsets
	) -> Result<()>,
) -> Result<()> {
	warn!("batch::run() called with more than one batch dimension");
	let dim = dims.next().unwrap();
	__run_recursive(dim, dims, offsets, &mut f)
}

#[inline]
pub fn run<const N: usize>(
	mut dims: MergedDimIter<N>, offsets: [usize; N],
	mut f: impl FnMut(
		usize,      // batch_size
		[usize; N], // batch_strides
		[usize; N], // offsets
	) -> Result<()>,
) -> Result<()> {
	match dims.len() {
		0 => f(1, [0; N], offsets),
		1 => {
			let first = dims.next().unwrap();
			f(first.size, first.strides, offsets)
		},
		_ => {
			cold_path();
			__run(dims, offsets, f)
		},
	}
}
