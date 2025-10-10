//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ffi::{c_int, c_void};
use std::hint::cold_path;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
#[repr(C)]
pub struct FfiSpan {
	pub ptr: *mut u8,
	pub len: usize,
}

#[repr(C)]
pub struct FfiBufferVMT {
	/// Returns span (data, len)
	pub span: unsafe extern "C" fn(this: *mut c_void) -> FfiSpan,

	/// Returns span (data, capacity)
	pub buf_span: unsafe extern "C" fn(this: *mut c_void) -> FfiSpan,

	/// Tries to reserve at least new_capacity. Returns buf_span(), i.e., (data, capacity).
	/// On success, the returned capacity will be >= new_capacity.
	///
	/// This will not deliberately over-allocate, but the allocator may still do so.
	/// Additionally, this function doesn't ever shrink the buffer.
	///
	/// If reallocation happens, only items up to len are preserved.
	pub reserve_exact: unsafe extern "C" fn(this: *mut c_void, new_capacity: usize) -> FfiSpan,

	/// Forces the length of the vector to new_len.
	/// On success (if new_len <= capacity), returns true.
	/// Otherwise, returns false and does nothing.
	///
	/// This function doesn't invalidate existing pointers into the buffer.
	pub set_len: unsafe extern "C" fn(this: *mut c_void, new_len: usize) -> c_int,
}

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct FfiBuffer<'a> {
	pub instance: *mut c_void,
	pub vmt: *const FfiBufferVMT,
	_marker: std::marker::PhantomData<&'a mut Vec<u8>>,
}

impl<'a> FfiBuffer<'a> {
	pub fn new(vec: &'a mut Vec<u8>) -> Self {
		extern "C" fn span(this: *mut c_void) -> FfiSpan {
			let this = this.cast::<Vec<u8>>();
			let this = unsafe { &mut *this };
			FfiSpan { ptr: this.as_mut_ptr(), len: this.len() }
		}

		extern "C" fn buf_span(this: *mut c_void) -> FfiSpan {
			let this = this.cast::<Vec<u8>>();
			let this = unsafe { &mut *this };
			FfiSpan {
				ptr: this.as_mut_ptr(),
				len: this.capacity(),
			}
		}

		extern "C" fn reserve_exact(this: *mut c_void, new_capacity: usize) -> FfiSpan {
			let this = this.cast::<Vec<u8>>();
			let this = unsafe { &mut *this };
			if new_capacity > this.capacity() {
				let additional = new_capacity - this.len();
				this.try_reserve_exact(additional);
			}
			FfiSpan {
				ptr: this.as_mut_ptr(),
				len: this.capacity(),
			}
		}

		extern "C" fn set_len(this: *mut c_void, new_len: usize) -> c_int {
			let this = this.cast::<Vec<u8>>();
			let this = unsafe { &mut *this };
			if new_len > this.capacity() {
				cold_path();
				return 0;
			}
			unsafe { this.set_len(new_len) };
			1
		}

		static VMT: FfiBufferVMT = FfiBufferVMT { span, buf_span, reserve_exact, set_len };

		Self {
			instance: std::ptr::from_mut(vec).cast(),
			vmt: std::ptr::addr_of!(VMT),
			_marker: std::marker::PhantomData,
		}
	}
}

//--------------------------------------------------------------------------------------------------
