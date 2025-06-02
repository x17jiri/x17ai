// Types of Buf:
// - Rc<BufOnDevice>
// - &BufOnDevice
// - &[T]
// Types of Map:
// - ND<4>
// - DynD

use std::hint::cold_path;

use crate::tensor::{DimIndex, SizeAndStride};

mod my_rc {
	use std::cell::Cell;
	use std::ptr::NonNull;

	#[repr(C)]
	pub struct RcInner<T> {
		rc: Cell<usize>,
		value: T,
	}

	pub struct Rc<T> {
		inner: NonNull<RcInner<T>>,
	}

	impl<T> Rc<T> {
		pub fn new(value: T) -> Self {
			Rc {
				inner: Box::leak(Box::new(RcInner { rc: Cell::new(1), value })).into(),
			}
		}

		pub fn from_inner(inner: &RcInner<T>) -> Self {
			inner.rc.set(inner.rc.get() + 1);
			Rc { inner: NonNull::from(inner) }
		}

		pub fn inner(&self) -> &RcInner<T> {
			unsafe { self.inner.as_ref() }
		}

		#[inline(never)]
		fn __drop(&mut self) {
			unsafe { std::ptr::drop_in_place(self.inner.as_mut()) };
		}
	}

	impl<T> Clone for Rc<T> {
		#[inline]
		fn clone(&self) -> Self {
			Rc::from_inner(self.inner())
		}
	}

	impl<T> Drop for Rc<T> {
		#[inline]
		fn drop(&mut self) {
			let inner = self.inner();
			inner.rc.set(inner.rc.get() - 1);
			if inner.rc.get() == 0 {
				Self::__drop(self);
			}
		}
	}

	impl<T> !std::marker::Send for Rc<T> {}
	impl<T> !std::marker::Sync for Rc<T> {}

	impl<T> std::ops::Deref for Rc<T> {
		type Target = T;

		#[inline]
		fn deref(&self) -> &Self::Target {
			&self.inner().value
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub trait DimRange {
	fn resolve(self, len: usize) -> std::ops::Range<usize>;
}

impl<I: DimIndex> DimRange for I {
	fn resolve(self, len: usize) -> std::ops::Range<usize> {
		let i = self.resolve(len);
		std::ops::Range { start: i, end: i + 1 }
	}
}

impl<I: DimIndex> DimRange for std::ops::Range<I> {
	fn resolve(self, len: usize) -> std::ops::Range<usize> {
		let start = self.start.resolve_range(len);
		let end = self.end.resolve_range(len);
		assert!(start <= end, "Invalid range: start ({}) > end ({})", start, end);
		std::ops::Range { start, end }
	}
}

impl<I: DimIndex> DimRange for std::ops::RangeInclusive<I> {
	fn resolve(self, len: usize) -> std::ops::Range<usize> {
		let start = self.start().resolve_range(len);
		let end = self.end().resolve(len) + 1;
		assert!(start <= end, "Invalid range: start ({}) > end ({})", start, end);
		std::ops::Range { start, end }
	}
}

impl<I: DimIndex> DimRange for std::ops::RangeFrom<I> {
	fn resolve(self, len: usize) -> std::ops::Range<usize> {
		let start = self.start.resolve_range(len);
		std::ops::Range { start, end: len }
	}
}

impl<I: DimIndex> DimRange for std::ops::RangeTo<I> {
	fn resolve(self, len: usize) -> std::ops::Range<usize> {
		let end = self.end.resolve_range(len);
		std::ops::Range { start: 0, end }
	}
}

impl<I: DimIndex> DimRange for std::ops::RangeToInclusive<I> {
	fn resolve(self, len: usize) -> std::ops::Range<usize> {
		let end = self.end.resolve(len) + 1;
		std::ops::Range { start: 0, end }
	}
}

impl DimRange for std::ops::RangeFull {
	fn resolve(self, len: usize) -> std::ops::Range<usize> {
		std::ops::Range { start: 0, end: len }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Tensor<Map: IMap, Buf: IBuffer> {
	pub buf: Buf,
	pub map: Map,
}

//--------------------------------------------------------------------------------------------------

pub trait IMap {
	fn ndim(&self) -> usize;

	fn dim_size<I: DimIndex>(&self, i: I) -> usize;
}

pub trait MergeDims<const M: usize> {
	type Output;

	fn merge_dims(self) -> Self::Output;
}

pub trait MergeAllDims {
	type Output;

	fn merge_all_dims(self) -> Self::Output;
}

pub trait ReshapeLastDim<const M: usize> {
	type Output;

	fn reshape_last_dim(self, to_shape: [usize; M]) -> Self::Output;
}

#[derive(Clone, Copy)]
pub struct ND<const N: usize> {
	pub dims: [SizeAndStride; N],
	pub offset: usize,
}

impl<const N: usize> IMap for ND<N> {
	fn ndim(&self) -> usize {
		N
	}

	fn dim_size<I: DimIndex>(&self, i: I) -> usize {
		self.dims[i.resolve(N)].size
	}
}

impl<const N: usize, const M: usize> MergeDims<M> for ND<N>
where
	[(); N-M]:,
	[(); N-M+1]:,
{
	type Output = Option<ND<{N-M+1}>>;

	fn merge_dims(self) -> Self::Output {
		let mut dims = [SizeAndStride::default(); N-M+1];
		for i in 0..N-M {
			dims[i] = self.dims[i];
		}
		let mut iter = self.dims[N-M..N].iter().copied().rev();
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
		dims[N-M] = merged;
		Some(ND {dims, offset: self.offset})
	}
}

impl<const N: usize> MergeAllDims for ND<N> {
	type Output = Option<ND<1>>;

	fn merge_all_dims(self) -> Self::Output {
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
		Some(ND {
			dims: [merged],
			offset: self.offset,
		})
	}
}

impl<const N: usize, const M: usize> ReshapeLastDim<M> for ND<N>
where
	[(); N-1]:,
	[(); N-1+M]:,
{
	type Output = Option<ND<{N-1+M}>>;

	fn reshape_last_dim(self, to_shape: [usize; M]) -> Self::Output {
		let mut dims = [SizeAndStride::default(); N-1+M];
		for i in 0..N-1 {
			dims[i] = self.dims[i];
		}
		let removed_dim = self.dims[N-1];
		let elems = to_shape.iter().copied().product::<usize>();
		if elems != removed_dim.size {
			cold_path();
			return None;
		}
		let mut stride = removed_dim.stride;
		for i in (N-1..N-1+M).rev() {
			let size = to_shape[i - (N-1)];
			dims[i] = SizeAndStride {
				size,
				stride,
			};
			stride *= size;
		}
		Some(ND { dims, offset: self.offset })
	}
}

//--------------------------------------------------------------------------------------------------

impl<I0: DimIndex, T> std::ops::Index<I0> for Tensor<ND<1>, &[T]> {
	type Output = T;

	fn index(&self, i0: I0) -> &Self::Output {
		let i0 = i0.resolve(self.map.dims[0].size);
		&self.buf[self.map.offset + i0 * self.map.dims[0].stride]
	}
}

impl<I0: DimIndex, T> std::ops::Index<(I0,)> for Tensor<ND<1>, &[T]> {
	type Output = T;

	fn index(&self, (i0,): (I0,)) -> &Self::Output {
		let i0 = i0.resolve(self.map.dims[0].size);
		&self.buf[self.map.offset + i0 * self.map.dims[0].stride]
	}
}

pub trait Slice1D {
	type Output;

	fn slice<R0: DimRange>(self, r0: R0) -> Self::Output;
}

impl<Buf: IBuffer> Slice1D for Tensor<ND<1>, Buf> {
	type Output = Tensor<ND<1>, Buf::Ref>;

	fn slice<R0: DimRange>(self, r0: R0) -> Self::Output {
		let r0 = r0.resolve(self.map.dims[0].size);
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

//--------------------------------------------------------------------------------------------------

impl<I0: DimIndex, I1: DimIndex, T> std::ops::Index<(I0, I1)> for Tensor<ND<2>, &[T]> {
	type Output = T;

	fn index(&self, (i0, i1): (I0, I1)) -> &Self::Output {
		let i0 = i0.resolve(self.map.dims[0].size);
		let i1 = i1.resolve(self.map.dims[1].size);
		&self.buf[self.map.offset + i0 * self.map.dims[0].stride + i1 * self.map.dims[1].stride]
	}
}

pub trait Slice2D {
	type Output;

	fn slice<R0: DimRange, R1: DimRange>(&self, r0: R0, r1: R1) -> Self::Output;
}

impl<Buf: IBuffer> Slice2D for Tensor<ND<2>, Buf> {
	type Output = Tensor<ND<2>, Buf::Ref>;

	fn slice<R0: DimRange, R1: DimRange>(&self, r0: R0, r1: R1) -> Self::Output {
		let r0 = r0.resolve(self.map.dims[0].size);
		let r1 = r1.resolve(self.map.dims[1].size);
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

//--------------------------------------------------------------------------------------------------

impl<I0: DimIndex, I1: DimIndex, I2: DimIndex, T> std::ops::Index<(I0, I1, I2)>
	for Tensor<ND<3>, &[T]>
{
	type Output = T;

	fn index(&self, (i0, i1, i2): (I0, I1, I2)) -> &Self::Output {
		let i0 = i0.resolve(self.map.dims[0].size);
		let i1 = i1.resolve(self.map.dims[1].size);
		let i2 = i2.resolve(self.map.dims[2].size);
		&self.buf[self.map.offset
			+ i0 * self.map.dims[0].stride
			+ i1 * self.map.dims[1].stride
			+ i2 * self.map.dims[2].stride]
	}
}

pub trait Slice3D {
	type Output;

	fn slice<R0: DimRange, R1: DimRange, R2: DimRange>(
		&self, r0: R0, r1: R1, r2: R2,
	) -> Self::Output;
}

impl<Buf: IBuffer> Slice3D for Tensor<ND<3>, Buf> {
	type Output = Tensor<ND<3>, Buf::Ref>;

	fn slice<R0: DimRange, R1: DimRange, R2: DimRange>(
		&self, r0: R0, r1: R1, r2: R2,
	) -> Self::Output {
		let r0 = r0.resolve(self.map.dims[0].size);
		let r1 = r1.resolve(self.map.dims[1].size);
		let r2 = r2.resolve(self.map.dims[2].size);
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

//--------------------------------------------------------------------------------------------------

pub trait IBuffer: Clone {
	type Ref: IBuffer;

	fn as_ref(&self) -> Self::Ref;
}

impl<T> IBuffer for &[T] {
	type Ref = Self;

	fn as_ref(&self) -> Self::Ref {
		*self
	}
}

//--------------------------------------------------------------------------------------------------
