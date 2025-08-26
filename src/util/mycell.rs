//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use core::borrow;
use std::cell::Cell;
use std::hint::cold_path;

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

//----Ref----------------------------------------------------------------------------------------------

/// This is my implementation of RefCell designed to be used with DeviceBuffer.
///
/// The thing about DeviceBuffer is that apart from data it also contains metadata. Things like
/// element count and dtype. Metadata is constant and I want to be able to access it without
/// borrowing.
///
/// So the following is possible without borrowing:
/// ```
/// let buf: Rc<mycell::RefCell<DeviceBuffer>> = ...;
/// let elems = buf.elems();
/// let dtype = buf.dtype();
/// ```
///
/// But to get to the actual data, we need either `borrow()` or `borrow_mut()`.
pub struct RefCell<T: ?Sized> {
	borrow_counter: Cell<isize>,
	value: T,
}

impl<T> RefCell<T> {
	pub fn new(value: T) -> Self {
		Self { borrow_counter: Cell::new(0), value }
	}
}

pub struct UnsafeBorrowFailFlag(isize);

impl Default for UnsafeBorrowFailFlag {
	fn default() -> Self {
		Self::new()
	}
}

impl UnsafeBorrowFailFlag {
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

pub struct UnsafeBorrowMutFailFlag(isize);

impl Default for UnsafeBorrowMutFailFlag {
	fn default() -> Self {
		Self::new()
	}
}

impl UnsafeBorrowMutFailFlag {
	pub fn new() -> Self {
		Self(0)
	}

	pub fn has_failed(&self) -> bool {
		self.0 != 0
	}

	pub fn check(&self) -> Result<(), BorrowMutError> {
		if self.has_failed() {
			cold_path();
			Err(BorrowMutError)
		} else {
			Ok(())
		}
	}
}

impl<T: ?Sized> RefCell<T> {
	pub unsafe fn unsafe_borrow(&self, fail: &mut UnsafeBorrowFailFlag) -> BorrowGuard<T> {
		let borrow_count = self.borrow_counter.get();
		fail.0 |= borrow_count;

		self.borrow_counter.set(borrow_count + 1);
		BorrowGuard { value: self }
	}

	pub fn try_borrow(&self) -> Result<BorrowGuard<T>, BorrowError> {
		let borrow_count = self.borrow_counter.get();
		if borrow_count < 0 {
			cold_path();
			Err(BorrowError)
		} else {
			self.borrow_counter.set(borrow_count + 1);
			Ok(BorrowGuard { value: self })
		}
	}

	pub unsafe fn unsafe_borrow_mut(
		&self,
		fail: &mut UnsafeBorrowMutFailFlag,
	) -> BorrowMutGuard<T> {
		let borrow_count = self.borrow_counter.get();
		fail.0 |= borrow_count;

		self.borrow_counter.set(borrow_count - 1);
		BorrowMutGuard { value: self }
	}

	pub fn try_borrow_mut(&self) -> Result<BorrowMutGuard<T>, BorrowMutError> {
		let borrow_count = self.borrow_counter.get();
		if borrow_count != 0 {
			cold_path();
			Err(BorrowMutError)
		} else {
			self.borrow_counter.set(borrow_count - 1);
			Ok(BorrowMutGuard { value: self })
		}
	}
}

impl<'a, T: ?Sized> std::ops::Deref for RefCell<T> {
	type Target = T;

	fn deref(&self) -> &T {
		&self.value
	}
}

//--------------------------------------------------------------------------------------------------

pub struct BorrowGuard<'a, T: ?Sized + 'a> {
	value: &'a RefCell<T>,
}

impl<'a, T: ?Sized + 'a> BorrowGuard<'a, T> {
	pub fn as_ref<'t>(&'t self) -> Ref<'t, T> {
		Ref { value: self.value }
	}
}

impl<'a, T: ?Sized> Drop for BorrowGuard<'a, T> {
	fn drop(&mut self) {
		let borrow_count = self.value.borrow_counter.get();
		debug_assert!(borrow_count > 0);
		self.value.borrow_counter.set(borrow_count - 1);
	}
}

impl<'a, T: ?Sized> std::ops::Deref for BorrowGuard<'a, T> {
	type Target = T;

	fn deref(&self) -> &T {
		&self.value.value
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Ref<'a, T: ?Sized + 'a> {
	value: &'a RefCell<T>,
}

impl<'a, T: ?Sized> std::ops::Deref for Ref<'a, T> {
	type Target = T;

	fn deref(&self) -> &T {
		&self.value.value
	}
}

//--------------------------------------------------------------------------------------------------

pub struct BorrowMutGuard<'a, T: ?Sized + 'a> {
	value: &'a RefCell<T>,
}

impl<'a, T: ?Sized + 'a> BorrowMutGuard<'a, T> {
	pub fn as_mut<'t>(&'t mut self) -> RefMut<'t, T> {
		RefMut { value: self.value }
	}
}

impl<'a, T: ?Sized> Drop for BorrowMutGuard<'a, T> {
	fn drop(&mut self) {
		let borrow_count = self.value.borrow_counter.get();
		debug_assert!(borrow_count < 0);
		self.value.borrow_counter.set(borrow_count + 1);
	}
}

impl<'a, T: ?Sized> std::ops::Deref for BorrowMutGuard<'a, T> {
	type Target = T;

	fn deref(&self) -> &T {
		&self.value.value
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RefMut<'a, T: ?Sized + 'a> {
	value: &'a RefCell<T>,
}

impl<'a, T: ?Sized> std::ops::Deref for RefMut<'a, T> {
	type Target = T;

	fn deref(&self) -> &T {
		&self.value.value
	}
}

//--------------------------------------------------------------------------------------------------
