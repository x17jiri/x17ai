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

pub struct SliceInfo<const N: usize, const I: usize, const S: bool> {
	pub items: [IndexOrRange; N],
	pub star_pos: usize,
}

impl SliceInfo<0, 0, false> {
	pub fn new() -> Self {
		Self { items: [], star_pos: 0 }
	}
}

pub trait AppendStar {
	type Output;
	fn append_star(self) -> Self::Output;
}

impl<const N: usize, const I: usize> AppendStar for SliceInfo<N, I, false> {
	type Output = SliceInfo<N, I, true>;

	fn append_star(self) -> Self::Output {
		Self::Output { items: self.items, star_pos: N }
	}
}

pub trait AppendDim<T> {
	type Output;
	fn append_dim(self, index: T) -> Self::Output;
}

impl<const N: usize, const I: usize, const S: bool> AppendDim<usize> for SliceInfo<N, I, S>
where
	[(); N + 1]:,
	[(); I + 1]:,
{
	type Output = SliceInfo<{ N + 1 }, { I + 1 }, S>;

	fn append_dim(self, index: usize) -> Self::Output {
		let index = IndexOrRange::Index(index);
		SliceInfo {
			items: concat_arrays(self.items, [index]),
			star_pos: self.star_pos,
		}
	}
}

impl<const N: usize, const I: usize, const S: bool> AppendDim<std::ops::Range<usize>>
	for SliceInfo<N, I, S>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SliceInfo<{ N + 1 }, I, S>;

	fn append_dim(self, range: std::ops::Range<usize>) -> Self::Output {
		let range = IndexOrRange::Range(Some(range.start)..Some(range.end));
		SliceInfo {
			items: concat_arrays(self.items, [range]),
			star_pos: self.star_pos,
		}
	}
}

impl<const N: usize, const I: usize, const S: bool> AppendDim<std::ops::RangeInclusive<usize>>
	for SliceInfo<N, I, S>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SliceInfo<{ N + 1 }, I, S>;

	fn append_dim(self, range: std::ops::RangeInclusive<usize>) -> Self::Output {
		let range = IndexOrRange::Range(Some(*range.start())..Some(*range.end() + 1));
		SliceInfo {
			items: concat_arrays(self.items, [range]),
			star_pos: self.star_pos,
		}
	}
}

impl<const N: usize, const I: usize, const S: bool> AppendDim<std::ops::RangeFrom<usize>>
	for SliceInfo<N, I, S>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SliceInfo<{ N + 1 }, I, S>;

	fn append_dim(self, range: std::ops::RangeFrom<usize>) -> Self::Output {
		let range = IndexOrRange::Range(Some(range.start)..None);
		SliceInfo {
			items: concat_arrays(self.items, [range]),
			star_pos: self.star_pos,
		}
	}
}

impl<const N: usize, const I: usize, const S: bool> AppendDim<std::ops::RangeTo<usize>>
	for SliceInfo<N, I, S>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SliceInfo<{ N + 1 }, I, S>;

	fn append_dim(self, range: std::ops::RangeTo<usize>) -> Self::Output {
		let range = IndexOrRange::Range(None..Some(range.end));
		SliceInfo {
			items: concat_arrays(self.items, [range]),
			star_pos: self.star_pos,
		}
	}
}

impl<const N: usize, const I: usize, const S: bool> AppendDim<std::ops::RangeToInclusive<usize>>
	for SliceInfo<N, I, S>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SliceInfo<{ N + 1 }, I, S>;

	fn append_dim(self, range: std::ops::RangeToInclusive<usize>) -> Self::Output {
		let range = IndexOrRange::Range(None..Some(range.end + 1));
		SliceInfo {
			items: concat_arrays(self.items, [range]),
			star_pos: self.star_pos,
		}
	}
}

impl<const N: usize, const I: usize, const S: bool> AppendDim<std::ops::RangeFull>
	for SliceInfo<N, I, S>
where
	[(); N + 1]:,
	[(); I]:,
{
	type Output = SliceInfo<{ N + 1 }, I, S>;

	fn append_dim(self, _range: std::ops::RangeFull) -> Self::Output {
		let range = IndexOrRange::Range(None..None);
		SliceInfo {
			items: concat_arrays(self.items, [range]),
			star_pos: self.star_pos,
		}
	}
}

#[macro_export]
macro_rules! s {
	($($rest:tt)*) => {
		$crate::s_append!($crate::tensor::generic::select::SliceInfo::new(), $($rest)*)
	};
}

// Helper macro for appending remaining items after ellipsis
#[macro_export]
macro_rules! s_append {
	($builder:expr, ) => {
		$builder
	};
	($builder:expr, *) => {
		$crate::tensor::generic::select::AppendStar::append_star($builder)
	};
	($builder:expr, *, $($rest:tt)*) => {
		$crate::s_append2!(
			$crate::tensor::generic::select::AppendStar::append_star($builder),
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
