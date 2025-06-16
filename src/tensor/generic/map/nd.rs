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
use crate::tensor::generic::map::{
	ElementsOverflowError, IndexOutOfBoundsError, MergeAllDimsError, NDShape, Narrow, NarrowError,
	ReshapeLastDimError, Select, SelectError, StrideCounter, StrideCounterUnchecked, merge_dims,
};
use crate::tensor::generic::universal_range::UniversalRange;
use crate::util::array;

#[derive(Clone, Copy)]
pub struct ND<const N: usize> {
	pub dims: [SizeAndStride; N],
	pub offset: usize,
}

impl<const N: usize> ND<N> {
	pub fn new(shape: &[usize; N]) -> Result<(Self, usize), ElementsOverflowError> {
		let mut stride_counter = StrideCounter::new();
		let dims = array::try_map_backward(shape, |_, &size| stride_counter.prepend_dim(size))?;
		let elems = stride_counter.elems();
		Ok((Self { dims, offset: 0 }, elems))
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

impl<const N: usize> TryFrom<&DD> for ND<N> {
	type Error = TryNDFromDDError;

	fn try_from(dd: &DD) -> std::result::Result<Self, TryNDFromDDError> {
		let dd_slice = dd.dims.as_slice();
		let Some(dims) = array::try_from_iter(dd_slice.iter().copied()) else {
			cold_path();
			return Err(TryNDFromDDError { nd_dims: N, dd_dims: dd.dims.len() });
		};
		Ok(Self { dims, offset: dd.offset })
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
	type Error = MergeAllDimsError;

	fn merge_dims(&self) -> Result<Self::Output, MergeAllDimsError> {
		let mut dims = [SizeAndStride::default(); N - M + 1];
		for i in 0..N - M {
			dims[i] = self.dims[i];
		}
		dims[N - M] = merge_dims(&self.dims[N - M..N])?;
		Ok(ND { dims, offset: self.offset })
	}
}

impl<const N: usize> MergeAllDims for ND<N> {
	type Output = ND<1>;
	type Error = MergeAllDimsError;

	fn merge_all_dims(&self) -> Result<Self::Output, MergeAllDimsError> {
		Ok(ND {
			dims: [merge_dims(&self.dims)?],
			offset: self.offset,
		})
	}
}

impl<const N: usize, const M: usize> ReshapeLastDim<M> for ND<N>
where
	[(); N - 1]:,
	[(); N - 1 + M]:,
{
	type Output = ND<{ N - 1 + M }>;
	type Error = ReshapeLastDimError; // TODO - we don't need InvalidNDim

	fn reshape_last_dim(&self, to_shape: [usize; M]) -> Result<Self::Output> {
		let last_dim = self.dims[N - 1];

		let elems = to_shape.iter().copied().product::<usize>();
		if elems != last_dim.size {
			cold_path();
			#[inline(never)]
			fn err_incompatible_reshape(removed_size: usize, to_shape: &[usize]) -> Error {
				format!(
					"Cannot reshape last dimension of size {removed_size} to shape {:?}.",
					to_shape
				)
				.into()
			}
			return Err(err_incompatible_reshape(last_dim.size, &to_shape));
		}

		let mut dims = [SizeAndStride::default(); N - 1 + M];
		for i in 0..N - 1 {
			dims[i] = self.dims[i];
		}
		let mut stride_counter = StrideCounterUnchecked::with_stride(last_dim.stride);
		for i in (N - 1..N - 1 + M).rev() {
			dims[i] = stride_counter.prepend_dim(to_shape[i - (N - 1)]);
		}

		Ok(ND { dims, offset: self.offset })
	}
}

impl<const N: usize> IndexToOffset<N> for ND<N> {
	fn index_to_offset(&self, index: [usize; N]) -> Result<usize, IndexOutOfBoundsError> {
		let mut offset = self.offset;
		for (d, (&i, &dim)) in index.iter().zip(self.dims.iter()).enumerate() {
			if i >= dim.size {
				cold_path();
				return Err(IndexOutOfBoundsError);
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
	type Error = SelectError;

	fn select(&self, dim: usize, index: usize) -> Result<Self::Output, SelectError> {
		if dim >= N {
			cold_path();
			return Err(SelectError::DimIndexOutOfBounds);
		}

		let removed_dim = &self.dims[dim];
		if index >= removed_dim.size {
			cold_path();
			return Err(SelectError::IndexOutOfBounds);
		}

		let mut dims = [Default::default(); N - 1];
		for i in 0..dim {
			dims[i] = self.dims[i];
		}
		for i in dim..N - 1 {
			dims[i] = self.dims[i + 1];
		}
		Ok(ND {
			dims,
			offset: self.offset + index * removed_dim.stride,
		})
	}

	unsafe fn select_unchecked(&self, dim: usize, index: usize) -> Self::Output {
		debug_assert!(dim < N);
		let removed_dim = self.dims.get_unchecked(dim);
		debug_assert!(index < removed_dim.size);
		let mut dims = [Default::default(); N - 1];
		for i in 0..dim {
			*dims.get_unchecked_mut(i) = *self.dims.get_unchecked(i);
		}
		for i in dim..N - 1 {
			*dims.get_unchecked_mut(i) = *self.dims.get_unchecked(i + 1);
		}
		ND {
			dims,
			offset: self.offset + index * removed_dim.stride,
		}
	}
}

impl<const N: usize> Narrow for ND<N> {
	type Output = Self;
	type Error = NarrowError;

	fn narrow(&self, _dim: usize, _range: UniversalRange) -> Result<Self::Output, NarrowError> {
		todo!("Narrow::narrow for ND<N> not implemented yet");
	}
}

impl<const N: usize> Transpose for ND<N> {
	type Output = Self;
	type Error = DimIndexOutOfBoundsError;

	fn transposed(mut self, d0: usize, d1: usize) -> Result<Self::Output, Self::Error> {
		if d0 >= N || d1 >= N {
			cold_path();
			return Err(DimIndexOutOfBoundsError);
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
