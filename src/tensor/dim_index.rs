//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

//------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DimIndexOutOfBoundsError;

impl std::error::Error for DimIndexOutOfBoundsError {}

impl std::fmt::Display for DimIndexOutOfBoundsError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Dimension index out of range.")
	}
}

//--------------------------------------------------------------------------------------------------

pub trait DimIndex: Copy {
	/// Allowed indexes are:
	///     0 ..< ndim
	///     -ndim ..= -1
	fn resolve_index(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError>;

	/// As opposed to indexes, range bounds can go up to `ndim`:
	///     0 ..<= ndim
	///     -ndim ..= -1
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError>;
}

impl DimIndex for usize {
	fn resolve_index(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError> {
		if self < ndim {
			Ok(self)
		} else {
			cold_path();
			Err(DimIndexOutOfBoundsError)
		}
	}
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError> {
		if self <= ndim {
			Ok(self)
		} else {
			cold_path();
			Err(DimIndexOutOfBoundsError)
		}
	}
}

impl DimIndex for isize {
	fn resolve_index(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError> {
		#[allow(clippy::cast_sign_loss)]
		let dim = if self >= 0 { self as usize } else { ndim.wrapping_add(self as usize) };
		if dim < ndim {
			Ok(dim)
		} else {
			cold_path();
			Err(DimIndexOutOfBoundsError)
		}
	}
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError> {
		#[allow(clippy::cast_sign_loss)]
		let dim = if self >= 0 { self as usize } else { ndim.wrapping_add(self as usize) };
		if dim <= ndim {
			Ok(dim)
		} else {
			cold_path();
			Err(DimIndexOutOfBoundsError)
		}
	}
}

impl DimIndex for u32 {
	fn resolve_index(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError> {
		(self as usize).resolve_index(ndim)
	}
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError> {
		(self as usize).resolve_range_bound(ndim)
	}
}

impl DimIndex for i32 {
	fn resolve_index(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError> {
		(self as isize).resolve_index(ndim)
	}
	fn resolve_range_bound(self, ndim: usize) -> Result<usize, DimIndexOutOfBoundsError> {
		(self as isize).resolve_range_bound(ndim)
	}
}

//--------------------------------------------------------------------------------------------------

pub trait DimRange {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, DimIndexOutOfBoundsError>;
}

impl<I: DimIndex> DimRange for I {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, DimIndexOutOfBoundsError> {
		let i = self.resolve_index(ndim)?;
		Ok(Range { start: i, end: i + 1 })
	}
}

impl<I: DimIndex> DimRange for Range<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, DimIndexOutOfBoundsError> {
		let start = self.start.resolve_range_bound(ndim)?;
		let end = self.end.resolve_range_bound(ndim)?.max(start);
		Ok(Range { start, end })
	}
}

impl<I: DimIndex> DimRange for RangeInclusive<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, DimIndexOutOfBoundsError> {
		let start = self.start().resolve_range_bound(ndim)?;
		let end = (self.end().resolve_index(ndim)? + 1).max(start);
		Ok(Range { start, end })
	}
}

impl<I: DimIndex> DimRange for RangeFrom<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, DimIndexOutOfBoundsError> {
		let start = self.start.resolve_range_bound(ndim)?;
		Ok(Range { start, end: ndim })
	}
}

impl<I: DimIndex> DimRange for RangeTo<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, DimIndexOutOfBoundsError> {
		let end = self.end.resolve_range_bound(ndim)?;
		Ok(Range { start: 0, end })
	}
}

impl<I: DimIndex> DimRange for RangeToInclusive<I> {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, DimIndexOutOfBoundsError> {
		let end = self.end.resolve_index(ndim)? + 1;
		Ok(Range { start: 0, end })
	}
}

impl DimRange for RangeFull {
	fn resolve_range(self, ndim: usize) -> Result<Range<usize>, DimIndexOutOfBoundsError> {
		Ok(Range { start: 0, end: ndim })
	}
}

//--------------------------------------------------------------------------------------------------
