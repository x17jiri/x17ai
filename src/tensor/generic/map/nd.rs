//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::convert::Infallible;
use std::hint::cold_path;

use super::{
	DD, IndexToOffset, Map, MergeAllDims, MergeDims, ReshapeLastDim, SizeAndStride, Transpose,
};
use crate::tensor::generic::dim_index::DimIndexOutOfBoundsError;
use crate::tensor::generic::map::{NDShape, Narrow, Select, init_strides};
use crate::tensor::generic::universal_range::UniversalRange;
use crate::util::array::try_array_from_iter;
use crate::{Error, Result};

#[derive(Clone, Copy)]
pub struct ND<const N: usize> {
	pub dims: [SizeAndStride; N],
	pub offset: usize,
}

impl<const N: usize> ND<N> {
	pub fn new(shape: &[usize; N]) -> Result<(Self, usize)> {
		let mut dims = shape.map(|size| SizeAndStride { size, stride: 0 });
		let elems = init_strides(&mut dims)?;
		let map = Self { dims, offset: 0 };
		Ok((map, elems))
	}
}

#[derive(Clone, Copy, Debug)]
pub struct TryNDFromDDError {
	pub nd_dims: usize,
	pub dd_dims: usize,
}

impl std::error::Error for TryNDFromDDError {}

impl std::fmt::Display for TryNDFromDDError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"Cannot convert from DD with {} dimensions to ND with {} dimensions.",
			self.dd_dims, self.nd_dims
		)
	}
}

impl<const N: usize> TryFrom<DD> for ND<N> {
	type Error = TryNDFromDDError;

	fn try_from(dyn_d: DD) -> std::result::Result<Self, TryNDFromDDError> {
		let Some(dims) = try_array_from_iter(dyn_d.dims.iter().copied()) else {
			cold_path();
			return Err(TryNDFromDDError { nd_dims: N, dd_dims: dyn_d.dims.len() });
		};
		Ok(Self { dims, offset: dyn_d.offset })
	}
}

impl<const N: usize> Map for ND<N> {
	fn ndim(&self) -> usize {
		N
	}

	fn size(&self, dim: usize) -> usize {
		if dim >= N {
			cold_path();
			return 1;
		}
		self.dims[dim].size
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
					#[cold]
					fn err_incompatible_strides() -> Error {
						"Cannot merge dimensions because of incompatible strides.".into()
					}
					return Err(err_incompatible_strides());
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
					#[cold]
					fn err_incompatible_strides() -> Error {
						"Cannot merge dimensions because of incompatible strides.".into()
					}
					return Err(err_incompatible_strides());
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
			#[cold]
			fn err_reshape_last_dim(removed_size: usize, to_shape: &[usize]) -> Error {
				format!(
					"Cannot reshape last dimension of size {removed_size} to shape {to_shape:?}. Total elements must match.",
				)
				.into()
			}
			return Err(err_reshape_last_dim(removed_dim.size, &to_shape));
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
				#[cold]
				fn err_index_out_of_range(i: usize, size: usize, d: usize) -> Error {
					format!("Index {i} out of range 0 ..< {size} for dimension {d}.").into()
				}
				return Err(err_index_out_of_range(i, dim.size, d));
			}
			offset += i * dim.stride;
		}
		Ok(offset)
	}
}

impl<const N: usize> Select for ND<N>
where
	[(); N - 1]:,
{
	type Output = ND<{ N - 1 }>;

	fn select(self, dim: usize, index: usize) -> Result<Self::Output> {
		if dim >= N {
			cold_path();
			return Err(DimIndexOutOfBoundsError.into());
		}
		let removed_dim = &self.dims[dim];
		if index >= removed_dim.size {
			#[cold]
			fn err_index_out_of_bounds(index: usize, size: usize) -> Error {
				format!("Index {index} is out of bounds for dimension of size {size}.").into()
			}
			return Err(err_index_out_of_bounds(index, removed_dim.size));
		}
		let mut new_dims = [Default::default(); N - 1];
		for i in 0..dim {
			new_dims[i] = self.dims[i];
		}
		for i in dim + 1..N {
			new_dims[i - 1] = self.dims[i];
		}
		Ok(ND {
			dims: new_dims,
			offset: self.offset + index * removed_dim.stride,
		})
	}

	unsafe fn select_unchecked(self, dim: usize, index: usize) -> Self::Output {
		debug_assert!(dim < N);
		let removed_dim = self.dims.get_unchecked(dim);
		debug_assert!(index < removed_dim.size);
		let mut new_dims = [Default::default(); N - 1];
		for i in 0..dim {
			*new_dims.get_unchecked_mut(i) = *self.dims.get_unchecked(i);
		}
		for i in dim + 1..N {
			*new_dims.get_unchecked_mut(i - 1) = *self.dims.get_unchecked(i);
		}
		ND {
			dims: new_dims,
			offset: self.offset + index * removed_dim.stride,
		}
	}
}

impl<const N: usize> Narrow for ND<N> {
	type Output = Self;

	fn narrow(self, dim: usize, range: UniversalRange) -> Result<Self::Output> {
		todo!("Narrow::narrow for ND<N> not implemented yet");
	}
}

impl<const N: usize> Transpose for ND<N> {
	type Output = Self;

	fn transposed(mut self, d0: usize, d1: usize) -> Result<Self> {
		if d0 >= N || d1 >= N {
			#[cold]
			fn err_transpose_out_of_range(d0: usize, d1: usize, N: usize) -> Error {
				format!("Cannot transpose dimensions {d0} and {d1}. Tensor has {N} dimensions.")
					.into()
			}
			return Err(err_transpose_out_of_range(d0, d1, N));
		}
		self.dims.swap(d0, d1);
		Ok(self)
	}
}

impl<const N: usize> NDShape<N> for ND<N> {
	type Error = Infallible;

	fn nd_shape(&self) -> std::result::Result<[usize; N], Self::Error> {
		let mut shape = [0; N];
		for (size, dim) in shape.iter_mut().zip(self.dims.iter()) {
			*size = dim.size;
		}
		Ok(shape)
	}
}
