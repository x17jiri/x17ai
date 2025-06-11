//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

pub struct UniversalRange {
	pub start: Option<usize>,
	pub end: Option<usize>,
}

impl From<usize> for UniversalRange {
	fn from(value: usize) -> Self {
		UniversalRange {
			start: Some(value), //
			end: Some(value + 1),
		}
	}
}

impl From<std::ops::Range<usize>> for UniversalRange {
	fn from(range: std::ops::Range<usize>) -> Self {
		UniversalRange {
			start: Some(range.start), //
			end: Some(range.end),
		}
	}
}

impl From<std::ops::RangeInclusive<usize>> for UniversalRange {
	fn from(range: std::ops::RangeInclusive<usize>) -> Self {
		UniversalRange {
			start: Some(*range.start()), //
			end: Some(*range.end()),
		}
	}
}

impl From<std::ops::RangeFrom<usize>> for UniversalRange {
	fn from(range: std::ops::RangeFrom<usize>) -> Self {
		UniversalRange {
			start: Some(range.start), //
			end: None,
		}
	}
}

impl From<std::ops::RangeTo<usize>> for UniversalRange {
	fn from(range: std::ops::RangeTo<usize>) -> Self {
		UniversalRange {
			start: None, //
			end: Some(range.end),
		}
	}
}

impl From<std::ops::RangeToInclusive<usize>> for UniversalRange {
	fn from(range: std::ops::RangeToInclusive<usize>) -> Self {
		UniversalRange {
			start: None, //
			end: Some(range.end + 1),
		}
	}
}

impl From<std::ops::RangeFull> for UniversalRange {
	fn from(_: std::ops::RangeFull) -> Self {
		UniversalRange {
			start: None, //
			end: None,
		}
	}
}
