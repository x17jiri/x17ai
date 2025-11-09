//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;

//--------------------------------------------------------------------------------------------------

pub trait IntrusiveRcTrait {
	unsafe fn inc_refcount(&self);
	unsafe fn dec_refcount(&self);
}

pub struct IntrusiveRc<T: IntrusiveRcTrait + ?Sized> {
	ptr: NonNull<T>,
}

impl<T: IntrusiveRcTrait + ?Sized> IntrusiveRc<T> {
	/// # Safety
	/// `ptr` must be a valid pointer to an object of type `T`.
	/// The object has to be properly initialized and refcount already incremented.
	/// This constructor will not increment the refcount, but `drop` will decrement it.
	pub unsafe fn new(ptr: NonNull<T>) -> Self {
		Self { ptr }
	}
}

impl<T: IntrusiveRcTrait + ?Sized> Clone for IntrusiveRc<T> {
	fn clone(&self) -> Self {
		unsafe {
			self.ptr.as_ref().inc_refcount();
		}
		Self { ptr: self.ptr }
	}
}

impl<T: IntrusiveRcTrait + ?Sized> Drop for IntrusiveRc<T> {
	fn drop(&mut self) {
		unsafe {
			self.ptr.as_ref().dec_refcount();
		}
	}
}

//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
