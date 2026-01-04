//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use super::{Device, DevicePtr};

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::DeviceAllocError;

//--------------------------------------------------------------------------------------------------

pub struct CPUBuffer {
	pub buffer: *mut [u8],
}

//--------------------------------------------------------------------------------------------------

pub struct CPUDevice {
	name: String,
}

impl CPUDevice {
	pub fn new() -> Rc<Self> {
		Self::new_named("CPU".to_string())
	}

	pub fn new_named(name: String) -> Rc<Self> {
		Rc::new(Self { name })
	}
}

const BUFFER_ALIGN: usize = std::mem::align_of::<u64>();

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn is_cpu(&self) -> bool {
		true
	}

	unsafe fn new_buffer(&self, bytes: usize) -> Result<DevicePtr, DeviceAllocError> {
		unsafe {
			if let struct_layout = std::alloc::Layout::new::<CPUBuffer>()
				&& let Ok(buffer_layout) = std::alloc::Layout::from_size_align(bytes, BUFFER_ALIGN)
				&& let Ok((layout, offset)) = struct_layout.extend(buffer_layout)
				&& let Some(memory) = NonNull::new(std::alloc::alloc(layout))
			{
				let header_ptr = memory.cast::<CPUBuffer>();
				let buffer_ptr = memory.add(offset);
				let buffer = std::ptr::slice_from_raw_parts_mut(buffer_ptr.as_ptr(), bytes);
				header_ptr.write(CPUBuffer { buffer });
				Ok(DevicePtr::new(header_ptr.as_ptr().cast()))
			} else {
				cold_path();
				Err(DeviceAllocError)
			}
		}
	}

	unsafe fn drop_buffer(&self, device_ptr: DevicePtr) {
		unsafe {
			let header_ptr = device_ptr.as_ptr::<CPUBuffer>();
			let buffer = (*header_ptr).buffer;
			let bytes = buffer.len();
			let struct_layout = std::alloc::Layout::new::<CPUBuffer>();
			let buffer_layout =
				std::alloc::Layout::from_size_align(bytes, BUFFER_ALIGN).unwrap_unchecked();
			let (layout, _offset) = struct_layout.extend(buffer_layout).unwrap_unchecked();
			let memory = header_ptr.cast::<u8>();
			std::alloc::dealloc(memory, layout);
		}
	}

	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: DevicePtr,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			let src: *const u8 = src.as_ptr();
			let header_ptr = dst.as_ptr::<CPUBuffer>();
			let buffer: *mut u8 = (*header_ptr).buffer.cast();
			std::ptr::copy_nonoverlapping(src, buffer, bytes);
			Ok(())
		}
	}

	unsafe fn download_data(
		&self,
		src: DevicePtr,
		dst: NonNull<u8>,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			let header_ptr = src.as_ptr::<CPUBuffer>();
			let buffer: *const u8 = (*header_ptr).buffer.cast();
			let dst: *mut u8 = dst.as_ptr();
			std::ptr::copy_nonoverlapping(buffer, dst, bytes);
			Ok(())
		}
	}
	/*
	unsafe fn run_fragment(
		&self,
		_compilation: &CompiledExpr,
		_fragment: FragmentIndex,
		_args: &KernelArgs,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("CPU device kernel execution is not implemented yet");
	}
	*/
}

//--------------------------------------------------------------------------------------------------
