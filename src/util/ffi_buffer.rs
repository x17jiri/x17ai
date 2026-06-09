//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ffi::c_void;
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

	/// Tries to reserve space for `additional` new elements.
	///
	/// On success, adds `additional` to `len` and returns a span of the extended area.
	/// The area is uninitialized.
	///
	/// On failure, returns a span with `ptr == null` and `len == 0`.
	///
	/// If reallocation happens, only items up to len are preserved.
	pub extend: unsafe extern "C" fn(this: *mut c_void, additional: usize) -> FfiSpan,

	/// Resets len to zero. Capacity is unchanged.
	pub clear: unsafe extern "C" fn(this: *mut c_void),

	/// Sets len to new_len. Assumes new_len <= capacity.
	pub set_len: unsafe extern "C" fn(this: *mut c_void, new_len: usize),
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

		extern "C" fn extend(this: *mut c_void, additional: usize) -> FfiSpan {
			let this = this.cast::<Vec<u8>>();
			let this = unsafe { &mut *this };
			if this.try_reserve(additional).is_err() {
				cold_path();
				return FfiSpan { ptr: std::ptr::null_mut(), len: 0 };
			}
			let ptr = unsafe { this.as_mut_ptr().add(this.len()) };
			unsafe { this.set_len(this.len() + additional) };
			FfiSpan { ptr, len: additional }
		}

		extern "C" fn clear(this: *mut c_void) {
			let this = this.cast::<Vec<u8>>();
			let this = unsafe { &mut *this };
			this.clear();
		}

		extern "C" fn set_len(this: *mut c_void, new_len: usize) {
			let this = this.cast::<Vec<u8>>();
			let this = unsafe { &mut *this };
			unsafe { this.set_len(new_len) };
		}

		static VMT: FfiBufferVMT = FfiBufferVMT { span, buf_span, extend, clear, set_len };

		Self {
			instance: std::ptr::from_mut(vec).cast(),
			vmt: std::ptr::addr_of!(VMT),
			_marker: std::marker::PhantomData,
		}
	}
}

//--------------------------------------------------------------------------------------------------
