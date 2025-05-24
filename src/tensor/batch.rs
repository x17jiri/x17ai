// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use super::TensorSize;
use super::dim_merger::{MergedDim, MergedDimIter};

// TODO - could this be implemented without recursion?
fn __run<const N: usize, F: FnMut(TensorSize, [TensorSize; N], [TensorSize; N])>(
	prev_dim: MergedDim<N>, mut dims: impl Clone + Iterator<Item = MergedDim<N>>,
	offsets: [TensorSize; N], f: &mut F,
) {
	if prev_dim.size > 1 {
		assert!(prev_dim.strides[0] > 0, "broadcast is disabled for this tensor");
	}

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

/// F: fn(batch_size: TensorSize, batch_strides: [TensorSize; N], offsets: [TensorSize; N])
pub fn run<'a, const N: usize, F: FnMut(TensorSize, [TensorSize; N], [TensorSize; N])>(
	dims: MergedDimIter<N>, offsets: [TensorSize; N], mut f: F,
) {
	let mut dims = dims.rev();
	if let Some(dim) = dims.next() {
		__run(dim, dims, offsets, &mut f);
	} else {
		f(1, [0; N], offsets);
	}
}
