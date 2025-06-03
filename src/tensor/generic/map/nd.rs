// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::hint::cold_path;

use crate::Error;

use super::{IndexToOffset, Map, MergeAllDims, MergeDims, ReshapeLastDim, SizeAndStride};

#[derive(Clone, Copy)]
pub struct ND<const N: usize> {
	pub dims: [SizeAndStride; N],
	pub offset: usize,
}

impl<const N: usize> Map for ND<N> {
	fn offset(&self) -> usize {
		self.offset
	}

	fn dims(&self) -> &[SizeAndStride] {
		&self.dims
	}

	fn dims_mut(&mut self) -> &mut [SizeAndStride] {
		&mut self.dims
	}
}

impl<const N: usize, const M: usize> MergeDims<M> for ND<N>
where
	[(); N - M]:,
	[(); N - M + 1]:,
{
	type Output = ND<{ N - M + 1 }>;

	fn merge_dims(self) -> Option<Self::Output> {
		let mut dims = [SizeAndStride::default(); N - M + 1];
		for i in 0..N - M {
			dims[i] = self.dims[i];
		}
		let mut iter = self.dims[N - M..N].iter().copied().rev();
		let mut merged = iter.next().unwrap_or(SizeAndStride { size: 1, stride: 1 });
		for dim in iter {
			if dim.stride == merged.size * merged.stride {
				merged.size *= dim.size;
			} else {
				cold_path();
				if dim.size == 1 {
					// Nothing to do
				} else if merged.size == 1 {
					merged = dim;
				} else if dim.size == 0 || merged.size == 0 {
					merged = SizeAndStride { size: 0, stride: 0 };
					break;
				} else {
					return None;
				}
			}
		}
		dims[N - M] = merged;
		Some(ND { dims, offset: self.offset })
	}
}

impl<const N: usize> MergeAllDims for ND<N> {
	type Output = ND<1>;

	fn merge_all_dims(self) -> Option<Self::Output> {
		let mut iter = self.dims.iter().copied().rev();
		let mut merged = iter.next().unwrap_or(SizeAndStride { size: 1, stride: 1 });
		for dim in iter {
			if dim.stride == merged.size * merged.stride {
				merged.size *= dim.size;
			} else {
				cold_path();
				if dim.size == 1 {
					// Nothing to do
				} else if merged.size == 1 {
					merged = dim;
				} else if dim.size == 0 || merged.size == 0 {
					merged = SizeAndStride { size: 0, stride: 0 };
					break;
				} else {
					return None;
				}
			}
		}
		Some(ND { dims: [merged], offset: self.offset })
	}
}

impl<const N: usize, const M: usize> ReshapeLastDim<M> for ND<N>
where
	[(); N - 1]:,
	[(); N - 1 + M]:,
{
	type Output = ND<{ N - 1 + M }>;

	fn reshape_last_dim(self, to_shape: [usize; M]) -> Option<Self::Output> {
		let mut dims = [SizeAndStride::default(); N - 1 + M];
		for i in 0..N - 1 {
			dims[i] = self.dims[i];
		}
		let removed_dim = self.dims[N - 1];
		let elems = to_shape.iter().copied().product::<usize>();
		if elems != removed_dim.size {
			cold_path();
			return None;
		}
		let mut stride = removed_dim.stride;
		for i in (N - 1..N - 1 + M).rev() {
			let size = to_shape[i - (N - 1)];
			dims[i] = SizeAndStride { size, stride };
			stride *= size;
		}
		Some(ND { dims, offset: self.offset })
	}
}

impl<const N: usize> IndexToOffset<N> for ND<N>
where
	[(); N - 1]:,
{
	fn index_to_offset(&self, index: [usize; N]) -> Result<usize, Error> {
		let mut offset = self.offset;
		for (d, (&i, &dim)) in index.iter().zip(self.dims.iter()).enumerate() {
			if i >= dim.size {
				return Err(format!(
					"Index {} out of range 0 ..< {} for dimension {}",
					i, dim.size, d
				)
				.into());
			}
			offset += i * dim.stride;
		}
		Ok(offset)
	}
}
