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
fn __run_recursive<const O: usize, const I: usize>(
	prev_dim: MergedDim<{ O + I }>,
	mut dims: impl Clone + Iterator<Item = MergedDim<{ O + I }>>,
	o_offsets: [usize; O],
	i_offsets: [usize; I],
	f: &mut impl FnMut(
		usize,      // batch_size
		[usize; O], // o_batch_strides
		[usize; I], // i_batch_strides
		[usize; O], // o_offsets
		[usize; I], // i_offsets
	) -> Result<()>,
) -> Result<()> {
	if prev_dim.size > 1 && prev_dim.strides[..O].iter().any(|&s| s == 0) {
		#[cold]
		fn err_cannot_broadcast_output_tensor() -> Error {
			"broadcast is disabled for output tensors".into()
		}
		return Err(err_cannot_broadcast_output_tensor());
	}

	if let Some(dim) = dims.next() {
		let mut o_offsets = o_offsets;
		let mut i_offsets = i_offsets;
		for _ in 0..prev_dim.size {
			__run_recursive(dim, dims.clone(), o_offsets, i_offsets, f)?;

			for j in 0..O {
				o_offsets[j] += prev_dim.strides[j];
			}
			for j in 0..I {
				i_offsets[j] += prev_dim.strides[O + j];
			}
		}
		Ok(())
	} else {
		let o_strides = std::array::from_fn(|i| prev_dim.strides[i]);
		let i_strides = std::array::from_fn(|i| prev_dim.strides[O + i]);
		f(prev_dim.size, o_strides, i_strides, o_offsets, i_offsets)
	}
}

#[inline(never)]
fn __run<const O: usize, const I: usize>(
	mut dims: MergedDimIter<{ O + I }>,
	o_offsets: [usize; O],
	i_offsets: [usize; I],
	mut f: impl FnMut(
		usize,      // batch_size
		[usize; O], // o_batch_strides
		[usize; I], // i_batch_strides
		[usize; O], // o_offsets
		[usize; I], // i_offsets
	) -> Result<()>,
) -> Result<()> {
	warn!("batch::run() called with more than one batch dimension");
	let dim = dims.next().unwrap();
	__run_recursive(dim, dims, o_offsets, i_offsets, &mut f)
}

#[inline]
pub fn run<const O: usize, const I: usize>(
	mut dims: MergedDimIter<{ O + I }>,
	o_offsets: [usize; O],
	i_offsets: [usize; I],
	mut f: impl FnMut(
		usize,      // batch_size
		[usize; O], // o_batch_strides
		[usize; I], // i_batch_strides
		[usize; O], // o_offsets
		[usize; I], // i_offsets
	) -> Result<()>,
) -> Result<()> {
	match dims.len() {
		0 => f(1, [0; O], [0; I], o_offsets, i_offsets),
		1 => {
			let first = dims.next().unwrap();
			let o_strides = std::array::from_fn(|i| first.strides[i]);
			let i_strides = std::array::from_fn(|i| first.strides[O + i]);
			f(first.size, o_strides, i_strides, o_offsets, i_offsets)
		},
		_ => {
			cold_path();
			__run(dims, o_offsets, i_offsets, &mut f)
		},
	}
}
