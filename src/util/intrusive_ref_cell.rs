//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;
use std::hint::cold_path;

//--------------------------------------------------------------------------------------------------

pub trait IntrusiveRefCellTrait: Sized {
	fn borrow_counter(&self) -> &BorrowCounter;

	fn borrow<'a>(&'a self, fail: &mut BorrowFailFlag) -> BorrowGuard<'a, Self> {
		BorrowGuard::new_branchless(self, fail)
	}

	fn try_borrow<'a>(&'a self) -> Result<BorrowGuard<'a, Self>, BorrowError> {
		BorrowGuard::new(self)
	}

	fn try_borrow_mut<'a>(
		&'a self,
		allowed_borrows: usize,
	) -> Result<BorrowMutGuard<'a, Self>, BorrowMutError> {
		BorrowMutGuard::new(self, allowed_borrows)
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BorrowError;

impl std::error::Error for BorrowError {}

impl std::fmt::Display for BorrowError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Cannot borrow the device buffer")
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BorrowMutError;

impl std::error::Error for BorrowMutError {}

impl std::fmt::Display for BorrowMutError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Cannot borrow the device buffer mutably")
	}
}

//--------------------------------------------------------------------------------------------------

/// This is my implementation of borrow checking designed to be used with DeviceBuffer.
///
/// There are several things specific about DeviceBuffer:
/// - apart from data, it also contains metadata. Things like element count.
/// Metadata is constant and I want to be able to access it without borrowing.
/// - There are situations where it is valid to have both mutable and immutable
/// borrows. For example, we want to allow expressions like `x = x + 1`
pub struct BorrowCounter {
	count: Cell<isize>,
}

impl BorrowCounter {
	pub fn new() -> Self {
		Self { count: Cell::new(0) }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct BorrowFailFlag(isize);

impl BorrowFailFlag {
	pub fn new() -> Self {
		Self(0)
	}

	pub fn has_failed(&self) -> bool {
		self.0 < 0
	}

	pub fn check(&self) -> Result<(), BorrowError> {
		if self.has_failed() {
			cold_path();
			Err(BorrowError)
		} else {
			Ok(())
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct BorrowGuard<'a, T: IntrusiveRefCellTrait> {
	value: &'a T,
}

impl<'a, T: IntrusiveRefCellTrait> BorrowGuard<'a, T> {
	pub fn new(value: &'a T) -> Result<Self, BorrowError> {
		let borrow_count = value.borrow_counter().count.get();
		if borrow_count < 0 {
			cold_path();
			Err(BorrowError)
		} else {
			value.borrow_counter().count.set(borrow_count + 1);
			Ok(BorrowGuard { value })
		}
	}

	pub fn new_branchless(value: &'a T, fail: &mut BorrowFailFlag) -> Self {
		let borrow_count = value.borrow_counter().count.get();
		fail.0 |= borrow_count;

		value.borrow_counter().count.set(borrow_count + 1);
		BorrowGuard { value }
	}
}

impl<'a, T: IntrusiveRefCellTrait> Drop for BorrowGuard<'a, T> {
	fn drop(&mut self) {
		let borrow_count = self.value.borrow_counter().count.get();
		debug_assert!(borrow_count > 0);
		self.value.borrow_counter().count.set(borrow_count - 1);
	}
}

impl<'a, T: IntrusiveRefCellTrait> std::ops::Deref for BorrowGuard<'a, T> {
	type Target = T;

	fn deref(&self) -> &T {
		self.value
	}
}

//--------------------------------------------------------------------------------------------------

pub struct BorrowMutGuard<'a, T: IntrusiveRefCellTrait> {
	value: &'a T,
	restore: isize,
}

impl<'a, T: IntrusiveRefCellTrait> BorrowMutGuard<'a, T> {
	pub fn new(value: &'a T, allowed_borrows: usize) -> Result<Self, BorrowMutError> {
		let allowed_borrows = allowed_borrows as isize;
		let borrow_count = value.borrow_counter().count.get();
		if borrow_count != allowed_borrows {
			cold_path();
			Err(BorrowMutError)
		} else {
			let restore = allowed_borrows + 1;
			value.borrow_counter().count.set(borrow_count - restore);
			Ok(BorrowMutGuard { value, restore })
		}
	}
}

impl<'a, T: IntrusiveRefCellTrait> Drop for BorrowMutGuard<'a, T> {
	fn drop(&mut self) {
		let borrow_count = self.value.borrow_counter().count.get();
		debug_assert!(borrow_count == -1);
		self.value.borrow_counter().count.set(borrow_count + self.restore);
	}
}

impl<'a, T: IntrusiveRefCellTrait> std::ops::Deref for BorrowMutGuard<'a, T> {
	type Target = T;

	fn deref(&self) -> &T {
		self.value
	}
}

//--------------------------------------------------------------------------------------------------
