//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::hint::cold_path;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;

//--------------------------------------------------------------------------------------------------

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

#[link(name = "dl")]
unsafe extern "C" {
	fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
	fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
	fn dlclose(handle: *mut c_void) -> c_int;
	fn dlerror() -> *const c_char;
}

fn clear_dlerror() {
	unsafe {
		dlerror();
	}
}

fn dlerror_detail() -> String {
	let err = unsafe { dlerror() };
	if err.is_null() {
		"No error details".into()
	} else {
		unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned()
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct DynamicLibrary {
	handle: NonNull<c_void>,
}

impl DynamicLibrary {
	pub fn open(path: &Path) -> Result<Self, String> {
		let Ok(path_cstr) = CString::new(path.as_os_str().as_bytes()) else {
			cold_path();
			let path = path.display();
			return Err(format!(
				"failed to load CUDA kernel library {path}; path contains an interior NUL byte"
			));
		};

		clear_dlerror();
		let handle = unsafe { dlopen(path_cstr.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
		let Some(handle) = NonNull::new(handle) else {
			cold_path();
			let path = path.display();
			let detail = dlerror_detail();
			return Err(format!("failed to load CUDA kernel library {path}: {detail}"));
		};

		Ok(Self { handle })
	}

	pub fn get_symbol(&self, symbol_name: &str) -> Result<*mut c_void, String> {
		let Ok(symbol_cstr) = CString::new(symbol_name) else {
			cold_path();
			return Err(format!(
				"failed to load CUDA kernel symbol {symbol_name:?}; symbol contains an interior NUL byte"
			));
		};

		clear_dlerror();
		let symbol = unsafe { dlsym(self.handle.as_ptr(), symbol_cstr.as_ptr()) };
		let Some(symbol) = NonNull::new(symbol) else {
			cold_path();
			let detail = dlerror_detail();
			return Err(format!("failed to load CUDA kernel symbol {symbol_name:?}: {detail}"));
		};

		Ok(symbol.as_ptr())
	}
}

impl Drop for DynamicLibrary {
	fn drop(&mut self) {
		unsafe {
			dlclose(self.handle.as_ptr());
		}
	}
}

//--------------------------------------------------------------------------------------------------
