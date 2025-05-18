// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::dim_merger::MergedDim;
use crate::*;
use smallvec::SmallVec;
use std::fmt;
use std::intrinsics::cold_path;

fn __run<const N: usize, F: FnMut(TensorSize, [TensorSize; N], [TensorSize; N])>(
	prev_dim: MergedDim<N>, mut dims: MergedDimIter<N>, offsets: [TensorSize; N],
	enable_broadcast: [bool; N], f: &mut F,
) {
	if prev_dim.size > 1 {
		for j in 0..N {
			assert!(enable_broadcast[j] || prev_dim.strides[j] > 0);
		}
	}

	if let Some(dim) = dims.next() {
		let mut offsets = offsets;
		for _ in 0..prev_dim.size {
			__run(dim, dims.clone(), offsets, enable_broadcast, f);

			for j in 0..N {
				offsets[j] += prev_dim.strides[j];
			}
		}
	} else {
		f(prev_dim.size, prev_dim.strides, offsets);
	}
}

/// F: fn(batch_size: TensorSize, batch_strides: [TensorSize; N], offsets: [TensorSize; N])
pub fn run<'a, const N: usize, F: FnMut(TensorSize, [TensorSize; N], [TensorSize; N])>(
	mut dims: MergedDimIter<N>, offsets: [TensorSize; N], enable_broadcast: [bool; N], mut f: F,
) {
	if let Some(dim) = dims.next() {
		__run(dim, dims, offsets, enable_broadcast, &mut f);
	} else {
		f(1, [0; N], offsets);
	}
}
