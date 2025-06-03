// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use super::Tensor;
use super::map::{IndexToOffset, Map};

//--------------------------------------------------------------------------------------------------
// Indexing

impl<const K: usize, M: Map + IndexToOffset<K>, T> std::ops::Index<[usize; K]> for Tensor<M, &[T]> {
	type Output = T;

	fn index(&self, index: [usize; K]) -> &Self::Output {
		let offset = self.map.index_to_offset(index).unwrap();
		&self.buf[offset]
	}
}

//--------------------------------------------------------------------------------------------------
// Slicing

/*
macro_rules! r {
	[$($range:expr),* $(,)?] => {
		[$(MyRange::from($range)),*]
	};
}

macro_rules! r {
	[$($range:expr),* $(,)?] => {
		[$(($range).into()),*]
	};
}
*/

pub trait Slice1D {
	type Output;

	fn slice<R0: DimRange>(self, r0: R0) -> Self::Output;
}

impl<B: Buffer> Slice1D for Tensor<ND<1>, B> {
	type Output = Tensor<ND<1>, B::Ref>;

	fn slice<R0: DimRange>(self, r0: R0) -> Self::Output {
		let dims = self.map.dims();
		let r0 = r0.resolve_range(dims[0].size).unwrap();
		let new_map = ND::<1> {
			dims: [SizeAndStride {
				size: r0.end - r0.start,
				stride: self.map.dims[0].stride,
			}],
			offset: self.map.offset + r0.start * self.map.dims[0].stride,
		};
		Tensor { buf: self.buf.as_ref(), map: new_map }
	}
}

pub trait Slice2D {
	type Output;

	fn slice<R0: DimRange, R1: DimRange>(&self, r0: R0, r1: R1) -> Self::Output;
}

impl<Buf: Buffer> Slice2D for Tensor<ND<2>, Buf> {
	type Output = Tensor<ND<2>, Buf::Ref>;

	fn slice<R0: DimRange, R1: DimRange>(&self, r0: R0, r1: R1) -> Self::Output {
		let r0 = r0.resolve_range(self.map.dims[0].size).unwrap();
		let r1 = r1.resolve_range(self.map.dims[1].size).unwrap();
		let new_map = ND::<2> {
			dims: [
				SizeAndStride {
					size: r0.end - r0.start,
					stride: self.map.dims[0].stride,
				},
				SizeAndStride {
					size: r1.end - r1.start,
					stride: self.map.dims[1].stride,
				},
			],
			offset: self.map.offset
				+ r0.start * self.map.dims[0].stride
				+ r1.start * self.map.dims[1].stride,
		};
		Tensor { buf: self.buf.as_ref(), map: new_map }
	}
}

pub trait Slice3D {
	type Output;

	fn slice<R0: DimRange, R1: DimRange, R2: DimRange>(
		&self, r0: R0, r1: R1, r2: R2,
	) -> Self::Output;
}

impl<Buf: Buffer> Slice3D for Tensor<ND<3>, Buf> {
	type Output = Tensor<ND<3>, Buf::Ref>;

	fn slice<R0: DimRange, R1: DimRange, R2: DimRange>(
		&self, r0: R0, r1: R1, r2: R2,
	) -> Self::Output {
		let r0 = r0.resolve_range(self.map.dims[0].size).unwrap();
		let r1 = r1.resolve_range(self.map.dims[1].size).unwrap();
		let r2 = r2.resolve_range(self.map.dims[2].size).unwrap();
		let new_map = ND::<3> {
			dims: [
				SizeAndStride {
					size: r0.end - r0.start,
					stride: self.map.dims[0].stride,
				},
				SizeAndStride {
					size: r1.end - r1.start,
					stride: self.map.dims[1].stride,
				},
				SizeAndStride {
					size: r2.end - r2.start,
					stride: self.map.dims[2].stride,
				},
			],
			offset: self.map.offset
				+ r0.start * self.map.dims[0].stride
				+ r1.start * self.map.dims[1].stride
				+ r2.start * self.map.dims[2].stride,
		};
		Tensor { buf: self.buf.as_ref(), map: new_map }
	}
}

pub trait Slice4D {
	type Output;

	fn slice<R0: DimRange, R1: DimRange, R2: DimRange, R3: DimRange>(
		&self, r0: R0, r1: R1, r2: R2, r3: R3,
	) -> Self::Output;
}

impl<Buf: Buffer> Slice4D for Tensor<ND<4>, Buf> {
	type Output = Tensor<ND<4>, Buf::Ref>;

	fn slice<R0: DimRange, R1: DimRange, R2: DimRange, R3: DimRange>(
		&self, r0: R0, r1: R1, r2: R2, r3: R3,
	) -> Self::Output {
		let r0 = r0.resolve_range(self.map.dims[0].size).unwrap();
		let r1 = r1.resolve_range(self.map.dims[1].size).unwrap();
		let r2 = r2.resolve_range(self.map.dims[2].size).unwrap();
		let r3 = r3.resolve_range(self.map.dims[3].size).unwrap();
		let new_map = ND::<4> {
			dims: [
				SizeAndStride {
					size: r0.end - r0.start,
					stride: self.map.dims[0].stride,
				},
				SizeAndStride {
					size: r1.end - r1.start,
					stride: self.map.dims[1].stride,
				},
				SizeAndStride {
					size: r2.end - r2.start,
					stride: self.map.dims[2].stride,
				},
				SizeAndStride {
					size: r3.end - r3.start,
					stride: self.map.dims[3].stride,
				},
			],
			offset: self.map.offset
				+ r0.start * self.map.dims[0].stride
				+ r1.start * self.map.dims[1].stride
				+ r2.start * self.map.dims[2].stride
				+ r3.start * self.map.dims[3].stride,
		};
		Tensor { buf: self.buf.as_ref(), map: new_map }
	}
}

//--------------------------------------------------------------------------------------------------
