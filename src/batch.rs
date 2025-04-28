// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use smallvec::SmallVec;
use std::fmt;

#[derive(Clone)]
pub struct Batch<const N: usize> {
	dim_merger: DimMerger<N>,
}

impl<const N: usize> Batch<N> {
	pub fn new<O: OutputHandler>(inputs: [&[SizeAndStride]; N], out: &mut O) -> Batch<N> {
		let mut merger = DimMerger::new_empty(ndim);

		// take the iterator to input dimensions, reverse it and append inf number of 1s
		let mut inputs: [_; N] = inputs.map(|i| {
			i.iter().rev().copied().chain(std::iter::repeat(SizeAndStride { size: 1, stride: 1 }))
		});

		for _ in 0..ndim {
			// Get sizes and strides for the current dimension from all inputs
			let mut in_dims = [Default::default(); N];
			for i in 0..N {
				in_dims[i] = inputs[i].next().unwrap();
			}

			// The dim_size should be the same for all inputs except for broadcasted inputs
			// So max() gets the dim_size of the non-broadcasted inputs.
			let dim_size = in_dims.iter().map(|i| i.size).max();
			let dim_size = dim_size.unwrap_or(1);
			let out_dim = out.prepend_dim(dim_size, true);

			// Get input strides. If needed, try to broadcast
			let in_strides = in_dims.map(|i| {
				if i.size == out_dim.size {
					i.stride
				} else {
					assert!(i.size == 1, "cannot broadcast: incompatible dimensions");
					0
				}
			});

			merger.prepend_dim(out_dim.size, out_dim.stride, in_strides);
		}

		Batch { dim_merger: merger }
	}

	fn __run<O: Copy, I: Copy, F: Fn(BatchBufOff<O>, [BatchBufOff<I>; N], usize)>(
		&self, mut out: BufOff<O>, mut in_: [BufOff<I>; N], f: &F, batch: &[BatchDim<N>],
	) {
		debug_assert!(!batch.is_empty());
		let batch_dim = unsafe { batch.get_unchecked(batch.len() - 1) };
		let sub_batch = unsafe { batch.get_unchecked(..batch.len() - 1) };
		if sub_batch.is_empty() {
			let out = out.make_batch(batch_dim.out_stride);
			let mut in_ = in_.map(|i| i.make_batch(0));
			for i in 0..N {
				in_[i].batch_stride = batch_dim.in_strides[i];
			}
			let batch_size = batch_dim.size;
			f(out, in_, batch_size);
		} else {
			for _ in 0..batch_dim.size {
				self.__run(out, in_, f, sub_batch);

				out.offset += batch_dim.out_stride;
				for i in 0..N {
					in_[i].offset += batch_dim.in_strides[i];
				}
			}
		}
	}

	pub fn run<O: Copy, I: Copy, F: Fn(BatchBufOff<O>, [BatchBufOff<I>; N], usize)>(
		&self, out: BufOff<O>, in_: [BufOff<I>; N], f: &F,
	) {
		if self.rev_dims.len() <= self.popped_dims {
			let out = out.make_batch(0);
			let in_ = in_.map(|i| i.make_batch(0));
			let batch_size = 1;
			f(out, in_, batch_size);
		} else {
			self.__run(out, in_, f, &self.rev_dims[self.popped_dims..]);
		}
	}
}
