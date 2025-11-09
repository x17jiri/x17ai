//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;
use std::ops::Deref;
use std::ptr::NonNull;

//--------------------------------------------------------------------------------------------------

pub struct RefCount {
	refcnt_minus_one: Cell<usize>,
}

impl RefCount {
	/// Creates a new RefCount with initial reference count of 1.
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self { refcnt_minus_one: Cell::new(0) }
	}

	pub fn inc_ref(&self) {
		let count = self.refcnt_minus_one.get();
		self.refcnt_minus_one.set(count + 1);
	}

	pub fn dec_ref(&self) {
		let count = self.refcnt_minus_one.get();
		self.refcnt_minus_one.set(count - 1);
	}

	pub fn has_single_ref(&self) -> bool {
		self.refcnt_minus_one.get() == 0
	}
}

//--------------------------------------------------------------------------------------------------

pub trait IntrusiveRcTrait {
	/// # Safety
	/// The returned RefCount must be handled correctly.
	unsafe fn refcount(&self) -> &RefCount;

	/// # Safety
	/// Can only be called when the refcount reaches zero.
	unsafe fn destroy(this: NonNull<Self>);

	fn has_single_ref(&self) -> bool {
		unsafe { self.refcount().has_single_ref() }
	}
}

pub struct IntrusiveRc<T: IntrusiveRcTrait> {
	ptr: NonNull<T>,
}

impl<T: IntrusiveRcTrait> IntrusiveRc<T> {
	/// # Safety
	/// `ptr` must be a valid pointer to an object of type `T`.
	/// The object has to be properly initialized and refcount already incremented.
	/// This constructor will not increment the refcount, but `drop` will decrement it.
	pub unsafe fn new(ptr: NonNull<T>) -> Self {
		Self { ptr }
	}

	pub unsafe fn from_box(value: Box<T>) -> Self {
		Self { ptr: NonNull::from(Box::leak(value)) }
	}
}

impl<T: IntrusiveRcTrait> Clone for IntrusiveRc<T> {
	fn clone(&self) -> Self {
		let rc = unsafe { self.ptr.as_ref().refcount() };
		rc.inc_ref();
		Self { ptr: self.ptr }
	}
}

impl<T: IntrusiveRcTrait> Drop for IntrusiveRc<T> {
	fn drop(&mut self) {
		let rc = unsafe { self.ptr.as_ref().refcount() };
		if rc.has_single_ref() {
			unsafe { T::destroy(self.ptr) };
		} else {
			rc.dec_ref();
		}
	}
}

impl<T> Deref for IntrusiveRc<T>
where
	T: IntrusiveRcTrait,
{
	type Target = T;

	fn deref(&self) -> &T {
		unsafe { self.ptr.as_ref() }
	}
}

//--------------------------------------------------------------------------------------------------
