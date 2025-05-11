// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::dim_merger::MergedDim;
use crate::*;
use smallvec::SmallVec;
use std::fmt;

fn __run<const N: usize, F: Fn(usize, [usize; N], [usize; N])>(
	prev_dim: MergedDim<N>, mut dims: MergedDimIter<N>, offsets: [usize; N], f: &F,
) {
	if let Some(dim) = dims.next() {
		let mut offsets = offsets;
		for _ in 0..prev_dim.size {
			__run(dim, dims.clone(), offsets, f);

			for j in 0..N {
				offsets[j] += prev_dim.strides[j];
			}
		}
	} else {
		f(prev_dim.size, prev_dim.strides, offsets);
	}
}

/// F: fn(batch_size: usize, batch_strides: [usize; N], offsets: [usize; N])
pub fn run<'a, const N: usize, F: Fn(usize, [usize; N], [usize; N])>(
	mut dims: MergedDimIter<N>, offsets: [usize; N], f: F,
) {
	if let Some(dim) = dims.next() {
		__run(dim, dims, offsets, &f);
	} else {
		f(1, [0; N], offsets);
	}
}
