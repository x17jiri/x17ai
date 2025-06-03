use std::hint::cold_path;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use super::Tensor;
use super::buffer::Buffer;
use super::map::{AllowIndex, Map, ND, SizeAndStride};

//--------------------------------------------------------------------------------------------------

pub trait DimIndex: Copy {
	/// Allowed indexes are:
	///     0 ..< ndim
	///     -ndim ..= -1
	fn resolve_index(self, ndim: usize) -> Result<usize, Error>;

	/// As opposed to indexes, range bounds can go up to `ndim`:
	///     0 ..<= ndim
	///     -ndim ..= -1
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, Error>;
}

impl DimIndex for usize {
	fn resolve_index(self, ndim: usize) -> Result<usize, Error> {
		if self < ndim {
			Ok(self)
		} else {
			cold_path();
			Err(format!("Dimension {} out of range 0 ..< {}", self, ndim).into())
		}
	}
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, Error> {
		if self <= ndim {
			Ok(self)
		} else {
			cold_path();
			Err(format!("Range bound out of range 0 ..<= {}", ndim).into())
		}
	}
}

impl DimIndex for isize {
	fn resolve_index(self, ndim: usize) -> Result<usize, Error> {
		let dim = if self >= 0 { self as usize } else { ndim.wrapping_add(self as usize) };
		if dim < ndim {
			Ok(dim)
		} else {
			cold_path();
			Err(format!("Dimension {} out of range -{} ..< {}", self, ndim, ndim).into())
		}
	}
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, Error> {
		let dim = if self >= 0 { self as usize } else { ndim.wrapping_add(self as usize) };
		if dim <= ndim {
			Ok(dim)
		} else {
			cold_path();
			Err(format!("Range bound out of range -{} ..<= {}", ndim, ndim).into())
		}
	}
}

impl DimIndex for u32 {
	fn resolve_index(self, ndim: usize) -> Result<usize, Error> {
		(self as usize).resolve_index(ndim)
	}
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, Error> {
		(self as usize).resolve_range_bound(ndim)
	}
}

impl DimIndex for i32 {
	fn resolve_index(self, ndim: usize) -> Result<usize, Error> {
		(self as isize).resolve_index(ndim)
	}
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, Error> {
		(self as isize).resolve_range_bound(ndim)
	}
}

//--------------------------------------------------------------------------------------------------

pub trait DimRange {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, Error>;
}

impl<I: DimIndex> DimRange for I {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, Error> {
		let i = self.resolve_index(ndim)?;
		Ok(Range { start: i, end: i + 1 })
	}
}

impl<I: DimIndex> DimRange for Range<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, Error> {
		let start = self.start.resolve_range_bound(ndim)?;
		let end = self.end.resolve_range_bound(ndim)?;
		if start > end {
			cold_path();
			return Err(format!("Range start {} is greater than end {}", start, end).into());
		}
		Ok(Range { start, end })
	}
}

impl<I: DimIndex> DimRange for RangeInclusive<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, Error> {
		let start = self.start().resolve_range_bound(ndim)?;
		let end = self.end().resolve_index(ndim)? + 1;
		if start > end {
			cold_path();
			return Err(format!("Range start {} is greater than end {}", start, end).into());
		}
		Ok(Range { start, end })
	}
}

impl<I: DimIndex> DimRange for RangeFrom<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, Error> {
		let start = self.start.resolve_range_bound(ndim)?;
		Ok(Range { start, end: ndim })
	}
}

impl<I: DimIndex> DimRange for RangeTo<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, Error> {
		let end = self.end.resolve_range_bound(ndim)?;
		Ok(Range { start: 0, end })
	}
}

impl<I: DimIndex> DimRange for RangeToInclusive<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, Error> {
		let end = self.end.resolve_index(ndim)? + 1;
		Ok(Range { start: 0, end })
	}
}

impl DimRange for RangeFull {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, Error> {
		Ok(Range { start: 0, end: ndim })
	}
}

//--------------------------------------------------------------------------------------------------
