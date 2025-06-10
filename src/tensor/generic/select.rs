//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::util::array::concat_arrays;

pub enum IndexOrRange {
	Index(usize),
	Range(std::ops::Range<Option<usize>>),
}

pub struct SelectionInfo<const N: usize, const I: usize, const E: bool> {
	pub items: [IndexOrRange; N],
	pub ellipsis_pos: usize,
}

impl SelectionInfo<0, 0, false> {
	pub fn new() -> Self {
		SelectionInfo { items: [], ellipsis_pos: 0 }
	}
}

pub trait AppendEllipsis {
	type Output;
	fn append_ellipsis(self) -> Self::Output;
}

impl<const N: usize, const I: usize> AppendEllipsis for SelectionInfo<N, I, false> {
	type Output = SelectionInfo<N, I, true>;

	fn append_ellipsis(self) -> Self::Output {
		SelectionInfo { items: self.items, ellipsis_pos: N }
	}
}

pub trait AppendDim<T> {
	type Output;
	fn append_dim(self, index: T) -> Self::Output;
}

impl<const N: usize, const I: usize, const E: bool> AppendDim<usize> for SelectionInfo<N, I, E>
where
	[(); N + 1]:,
	[(); I + 1]:,
{
	type Output = SelectionInfo<{ N + 1 }, { I + 1 }, E>;

	fn append_dim(self, index: usize) -> Self::Output {
		let index = IndexOrRange::Index(index);
		SelectionInfo {
			items: concat_arrays(self.items, [index]),
			ellipsis_pos: self.ellipsis_pos,
		}
	}
}

impl<const N: usize, const I: usize, const E: bool> AppendDim<std::ops::Range<usize>>
	for SelectionInfo<N, I, E>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SelectionInfo<{ N + 1 }, I, E>;

	fn append_dim(self, range: std::ops::Range<usize>) -> Self::Output {
		let range = IndexOrRange::Range(Some(range.start)..Some(range.end));
		SelectionInfo {
			items: concat_arrays(self.items, [range]),
			ellipsis_pos: self.ellipsis_pos,
		}
	}
}

impl<const N: usize, const I: usize, const E: bool> AppendDim<std::ops::RangeInclusive<usize>>
	for SelectionInfo<N, I, E>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SelectionInfo<{ N + 1 }, I, E>;

	fn append_dim(self, range: std::ops::RangeInclusive<usize>) -> Self::Output {
		let range = IndexOrRange::Range(Some(*range.start())..Some(*range.end() + 1));
		SelectionInfo {
			items: concat_arrays(self.items, [range]),
			ellipsis_pos: self.ellipsis_pos,
		}
	}
}

impl<const N: usize, const I: usize, const E: bool> AppendDim<std::ops::RangeFrom<usize>>
	for SelectionInfo<N, I, E>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SelectionInfo<{ N + 1 }, I, E>;

	fn append_dim(self, range: std::ops::RangeFrom<usize>) -> Self::Output {
		let range = IndexOrRange::Range(Some(range.start)..None);
		SelectionInfo {
			items: concat_arrays(self.items, [range]),
			ellipsis_pos: self.ellipsis_pos,
		}
	}
}

impl<const N: usize, const I: usize, const E: bool> AppendDim<std::ops::RangeTo<usize>>
	for SelectionInfo<N, I, E>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SelectionInfo<{ N + 1 }, I, E>;

	fn append_dim(self, range: std::ops::RangeTo<usize>) -> Self::Output {
		let range = IndexOrRange::Range(None..Some(range.end));
		SelectionInfo {
			items: concat_arrays(self.items, [range]),
			ellipsis_pos: self.ellipsis_pos,
		}
	}
}

impl<const N: usize, const I: usize, const E: bool> AppendDim<std::ops::RangeToInclusive<usize>>
	for SelectionInfo<N, I, E>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SelectionInfo<{ N + 1 }, I, E>;

	fn append_dim(self, range: std::ops::RangeToInclusive<usize>) -> Self::Output {
		let range = IndexOrRange::Range(None..Some(range.end + 1));
		SelectionInfo {
			items: concat_arrays(self.items, [range]),
			ellipsis_pos: self.ellipsis_pos,
		}
	}
}

impl<const N: usize, const I: usize, const E: bool> AppendDim<std::ops::RangeFull>
	for SelectionInfo<N, I, E>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SelectionInfo<{ N + 1 }, I, E>;

	fn append_dim(self, _range: std::ops::RangeFull) -> Self::Output {
		let range = IndexOrRange::Range(None..None);
		SelectionInfo {
			items: concat_arrays(self.items, [range]),
			ellipsis_pos: self.ellipsis_pos,
		}
	}
}

#[macro_export]
macro_rules! sel {
	($($rest:tt)*) => {
		$crate::s_append!($crate::tensor::generic::select::SelectionInfo::new(), $($rest)*)
	};
}

// Helper macro for appending remaining items after ellipsis
#[macro_export]
macro_rules! s_append {
	($builder:expr, ) => {
		$builder
	};
	($builder:expr, *) => {
		$crate::tensor::generic::select::AppendEllipsis::append_ellipsis($builder)
	};
	($builder:expr, *, $($rest:tt)*) => {
		$crate::s_append2!(
			$crate::tensor::generic::select::AppendEllipsis::append_ellipsis($builder),
			$($rest)*
		)
	};
	($builder:expr, $e:expr) => {
		$crate::tensor::generic::select::AppendDim::append_dim($builder, $e)
	};
	($builder:expr, $e:expr, $($rest:tt)*) => {
		$crate::s_append!(
			$crate::tensor::generic::select::AppendDim::append_dim($builder, $e),
			$($rest)*
		)
	};
}

#[macro_export]
macro_rules! s_append2 {
	($builder:expr $(,)?) => {
		$builder
	};
	($builder:expr, $e:expr) => {
		$crate::tensor::generic::select::AppendDim::append_dim($builder, $e)
	};
	($builder:expr, $e:expr, $($rest:tt)*) => {
		$crate::s_append2!(
			$crate::tensor::generic::select::AppendDim::append_dim($builder, $e),
			$($rest)*
		)
	};
}
