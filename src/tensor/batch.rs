// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::intrinsics::cold_path;

use log::warn;

use super::TensorSize;
use super::dim_merger::{MergedDim, MergedDimIter};

#[inline(never)]
fn __run_recursive<const N: usize, F: FnMut(TensorSize, [TensorSize; N], [TensorSize; N])>(
	prev_dim: MergedDim<N>, mut dims: impl Clone + Iterator<Item = MergedDim<N>>,
	offsets: [TensorSize; N], f: &mut F,
) {
	if prev_dim.size > 1 {
		assert!(prev_dim.strides[0] > 0, "broadcast is disabled for this tensor");
	}

	if let Some(dim) = dims.next() {
		let mut offsets = offsets;
		for _ in 0..prev_dim.size {
			__run_recursive(dim, dims.clone(), offsets, f);

			for j in 0..N {
				offsets[j] += prev_dim.strides[j];
			}
		}
	} else {
		f(prev_dim.size, prev_dim.strides, offsets);
	}
}

#[inline(never)]
fn __run<'a, const N: usize, F: FnMut(TensorSize, [TensorSize; N], [TensorSize; N])>(
	mut dims: MergedDimIter<N>, offsets: [TensorSize; N], mut f: F,
) {
	warn!("batch::run() called with more than one batch dimension");
	let dim = dims.next().unwrap();
	__run_recursive(dim, dims, offsets, &mut f);
}

/// F: fn(batch_size: TensorSize, batch_strides: [TensorSize; N], offsets: [TensorSize; N])
#[inline]
pub fn run<'a, const N: usize, F: FnMut(TensorSize, [TensorSize; N], [TensorSize; N])>(
	mut dims: MergedDimIter<N>, offsets: [TensorSize; N], mut f: F,
) {
	match dims.len() {
		0 => f(1, [0; N], offsets),
		1 => {
			let first = dims.next().unwrap();
			f(first.size, first.strides, offsets);
		},
		_ => {
			cold_path();
			__run(dims, offsets, f);
		},
	}
}
