pub mod buffer;
pub mod dim_index;
pub mod elem_index;
pub mod map;

// Types of Buf:
// - Rc<BufOnDevice>
// - &BufOnDevice
// - &[T]
// Types of Map:
// - ND<4>
// - DynD

use buffer::Buffer;
use dim_index::DimIndex;
use map::{Map, MergeAllDims, MergeDims, ReshapeLastDim};

pub use dim_index::{Slice1D, Slice2D, Slice3D, Slice4D};

//--------------------------------------------------------------------------------------------------
// Tensor

pub struct Tensor<M: Map, B: Buffer> {
	pub map: M,
	pub buf: B,
}

impl<M: Map, B: Buffer> Tensor<M, B> {
	pub fn ndim(&self) -> usize {
		self.map.dims().len()
	}

	pub fn elems(&self) -> usize {
		self.map.dims().iter().map(|dim| dim.size).product()
	}

	pub fn merge_dims<const K: usize>(self) -> Option<Tensor<M::Output, B>>
	where
		M: MergeDims<K>,
	{
		let new_map = self.map.merge_dims()?;
		Some(Tensor { buf: self.buf, map: new_map })
	}

	pub fn merge_all_dims(self) -> Option<Tensor<M::Output, B>>
	where
		M: MergeAllDims,
	{
		let new_map = self.map.merge_all_dims()?;
		Some(Tensor { buf: self.buf, map: new_map })
	}

	pub fn reshape_last_dim<const K: usize>(
		self, to_shape: [usize; K],
	) -> Option<Tensor<M::Output, B>>
	where
		M: ReshapeLastDim<K>,
	{
		let new_map = self.map.reshape_last_dim(to_shape)?;
		Some(Tensor { buf: self.buf, map: new_map })
	}

	pub fn transposed<D1: DimIndex, D2: DimIndex>(self, dim1: D1, dim2: D2) -> Self {
		let ndim = self.ndim();
		let dim1 = dim1.resolve_index(ndim);
		let dim2 = dim2.resolve_index(ndim);
		let Tensor { buf, mut map } = self;
		let dims = map.dims_mut();
		dims.swap(dim1, dim2);
		Tensor { buf, map }
	}
}

//--------------------------------------------------------------------------------------------------
