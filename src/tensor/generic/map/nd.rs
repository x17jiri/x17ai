//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use super::{
	CompactND, DynD, IndexToOffset, Map, MergeAllDims, MergeDims, ReshapeLastDim, SizeAndStride,
	Transpose,
};
use crate::{Error, Result};

#[derive(Clone, Copy)]
pub struct ND<const N: usize> {
	pub dims: [SizeAndStride; N],
	pub offset: usize,
}

impl<const N: usize> From<CompactND<N>> for ND<N>
where
	[(); N - 2]:,
{
	fn from(compact: CompactND<N>) -> Self {
		let mut dims = [SizeAndStride::default(); N];
		let mut stride = 1;
		for i in (1..N).rev() {
			let size = compact.shape[i];
			dims[i] = SizeAndStride { size, stride };
			stride *= size;
		}
		dims[0] = SizeAndStride {
			size: compact.shape[0],
			stride: compact.outer_stride,
		};
		ND { dims, offset: compact.offset }
	}
}

impl<const N: usize> TryFrom<DynD> for ND<N> {
	type Error = Error;

	fn try_from(dyn_d: DynD) -> Result<Self> {
		if dyn_d.dims.len() != N {
			cold_path();
			return Err(format!(
				"Cannot convert DynD with {} dimensions to ND with {} dimensions.",
				dyn_d.dims.len(),
				N
			)
			.into());
		}
		let mut dims = [SizeAndStride::default(); N];
		for (i, dim) in dyn_d.dims.iter().enumerate() {
			dims[i] = *dim;
		}
		Ok(ND { dims, offset: dyn_d.offset })
	}
}

impl<const N: usize> Map for ND<N> {
	fn ndim(&self) -> usize {
		N
	}

	fn elems(&self) -> usize {
		self.dims.iter().map(|dim| dim.size).product()
	}

	fn span(&self) -> std::ops::Range<usize> {
		let start = self.offset;
		if self.elems() == 0 {
			cold_path();
			return start..start;
		}
		let len = self.dims.iter().map(|dim| (dim.size - 1) * dim.stride).sum::<usize>() + 1;
		let end = start + len;
		start..end
	}

	fn is_contiguous(&self) -> bool {
		// TODO - this calculation doesn't work. Or does it?????
		// Consider these dims:
		//     dim 0: size 3, stride 10
		//     dim 1: size 10, stride 0
		//
		//     len = 2*10 + 9*0 + 1 = 21
		let elems = self.elems();
		if elems == 0 {
			cold_path();
			return true;
		}
		let len = self.dims.iter().map(|dim| (dim.size - 1) * dim.stride).sum::<usize>() + 1;
		len == elems
	}
}

impl<const N: usize, const M: usize> MergeDims<M> for ND<N>
where
	[(); N - M]:,
	[(); N - M + 1]:,
{
	type Output = ND<{ N - M + 1 }>;

	fn merge_dims(self) -> Result<Self::Output> {
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
					return Err(
						format!("Cannot merge dimensions because of incompatible strides").into()
					);
				}
			}
		}
		dims[N - M] = merged;
		Ok(ND { dims, offset: self.offset })
	}
}

impl<const N: usize> MergeAllDims for ND<N> {
	type Output = ND<1>;

	fn merge_all_dims(self) -> Result<Self::Output> {
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
					return Err(
						format!("Cannot merge dimensions because of incompatible strides").into()
					);
				}
			}
		}
		Ok(ND { dims: [merged], offset: self.offset })
	}
}

impl<const N: usize, const M: usize> ReshapeLastDim<M> for ND<N>
where
	[(); N - 1]:,
	[(); N - 1 + M]:,
{
	type Output = ND<{ N - 1 + M }>;

	fn reshape_last_dim(self, to_shape: [usize; M]) -> Result<Self::Output> {
		let mut dims = [SizeAndStride::default(); N - 1 + M];
		for i in 0..N - 1 {
			dims[i] = self.dims[i];
		}
		let removed_dim = self.dims[N - 1];
		let elems = to_shape.iter().copied().product::<usize>();
		if elems != removed_dim.size {
			cold_path();
			return Err(format!(
				"Cannot reshape last dimension of size {} to shape {:?}. Total elements must match.",
				removed_dim.size, to_shape
			)
			.into());
		}
		let mut stride = removed_dim.stride;
		for i in (N - 1..N - 1 + M).rev() {
			let size = to_shape[i - (N - 1)];
			dims[i] = SizeAndStride { size, stride };
			stride *= size;
		}
		Ok(ND { dims, offset: self.offset })
	}
}

impl<const N: usize> IndexToOffset<N> for ND<N>
where
	[(); N - 1]:,
{
	fn index_to_offset(&self, index: [usize; N]) -> Result<usize> {
		let mut offset = self.offset;
		for (d, (&i, &dim)) in index.iter().zip(self.dims.iter()).enumerate() {
			if i >= dim.size {
				cold_path();
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

impl<const N: usize> Transpose for ND<N> {
	type Output = Self;

	fn transposed(mut self, d0: usize, d1: usize) -> Result<Self> {
		if d0 >= N || d1 >= N {
			return Err(format!(
				"Cannot transpose dimension {} with {}. Tensor has {} dimensions.",
				d0, d1, N
			)
			.into());
		}
		self.dims.swap(d0, d1);
		Ok(self)
	}
}
