//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ErrPack;
use crate::new::expr::compilation::{Compilation, FragmentIndex};
use crate::tensor::TensorOpError;

//--------------------------------------------------------------------------------------------------

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DevicePtr {
	ptr: *mut (),
}

impl DevicePtr {
	#[inline]
	pub fn new(ptr: *mut ()) -> Self {
		Self { ptr }
	}

	/// # Safety
	/// The pointer should only be used by device-specific code that knows what the pointer is.
	///
	/// It could be the device pointer casted to host pointer, or even some handle
	/// that can be used to get the real device pointer.
	///
	/// Since it can be a casted device pointer, dereferencing it may be undefined behavior.
	///
	/// And since it can be some handle, pointer arithmetic on it may not make sense.
	#[inline]
	pub unsafe fn as_ptr<T>(&self) -> *mut T {
		self.ptr.cast::<T>()
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceAllocError;

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct KernelArg {
	pub stride_bytes: [usize; 3],
	pub buf: DevicePtr,
}

#[repr(C)]
pub struct KernelOutput {
	pub buf: DevicePtr,
}

#[repr(C)]
pub struct KernelArgs {
	shape: [usize; 3],
	extra_memory: usize,
	arg_count: usize,
	output_count: usize,
	args: [MaybeUninit<KernelArg>; 0],
	outputs: [MaybeUninit<KernelOutput>; 0],
}

impl KernelArgs {
	pub fn set_shape(&mut self, shape: [usize; 3]) {
		self.shape = shape;
	}

	pub fn args_mut(&mut self) -> &mut [MaybeUninit<KernelArg>] {
		unsafe {
			let arg_count = self.arg_count;
			let struct_layout = std::alloc::Layout::new::<Self>();
			let args_layout = std::alloc::Layout::array::<MaybeUninit<KernelArg>>(self.arg_count)
				.unwrap_unchecked();
			let (_layout, args_offset) = struct_layout.extend(args_layout).unwrap_unchecked();
			#[allow(clippy::cast_ptr_alignment)]
			let args_ptr = std::ptr::from_mut(self)
				.cast::<u8>()
				.add(args_offset)
				.cast::<MaybeUninit<KernelArg>>();
			std::slice::from_raw_parts_mut(args_ptr, arg_count)
		}
	}

	pub fn outputs_mut(&mut self) -> &mut [MaybeUninit<KernelOutput>] {
		unsafe {
			let arg_count = self.arg_count;
			let output_count = self.output_count;
			let struct_layout = std::alloc::Layout::new::<Self>();
			let args_layout =
				std::alloc::Layout::array::<MaybeUninit<KernelArg>>(arg_count).unwrap_unchecked();
			let outputs_layout =
				std::alloc::Layout::array::<MaybeUninit<KernelOutput>>(output_count)
					.unwrap_unchecked();
			let (layout, _args_offset) = struct_layout.extend(args_layout).unwrap_unchecked();
			let (_layout, outputs_offset) = layout.extend(outputs_layout).unwrap_unchecked();
			#[allow(clippy::cast_ptr_alignment)]
			let outputs_ptr = std::ptr::from_mut(self)
				.cast::<u8>()
				.add(outputs_offset)
				.cast::<MaybeUninit<KernelOutput>>();
			std::slice::from_raw_parts_mut(outputs_ptr, output_count)
		}
	}

	pub fn extra_memory(arg_count: usize, output_count: usize) -> usize {
		const {
			assert!(std::mem::align_of::<KernelArg>() <= std::mem::align_of::<KernelArgs>());
			assert!(std::mem::align_of::<KernelOutput>() <= std::mem::align_of::<KernelArg>());
		}
		arg_count * std::mem::size_of::<KernelArg>()
			+ output_count * std::mem::size_of::<KernelOutput>()
	}

	pub fn set_counts(&mut self, arg_count: usize, output_count: usize) -> Reuslt<(), ()> {
		if self.extra_memory >= Self::extra_memory(arg_count, output_count) {
			self.arg_count = arg_count;
			self.output_count = output_count;
			Ok(())
		} else {
			cold_path();
			Err(())
		}
	}
}

#[repr(transparent)]
pub struct KernelArgsBox {
	pub ptr: NonNull<KernelArgs>,
}

impl KernelArgsBox {
	/// # Panics
	/// Panics if allocation fails.
	pub fn new(extra_memory: usize) -> Self {
		let layout = std::alloc::Layout::from_size_align(
			std::mem::size_of::<KernelArgs>() + extra_memory,
			std::mem::align_of::<KernelArgs>(),
		);
		#[allow(clippy::panic)]
		if let Ok(layout) = layout
			&& let Some(raw_ptr) = NonNull::new(unsafe { std::alloc::alloc(layout) })
		{
			let ptr = raw_ptr.cast();
			unsafe {
				ptr.write(KernelArgs {
					shape: [0; 3],
					extra_memory,
					arg_count: 0,
					output_count: 0,
					args: [],
					outputs: [],
				});
			}
			Self { ptr }
		} else {
			cold_path();
			panic!("allocation error");
		}
	}
}

impl Drop for KernelArgsBox {
	fn drop(&mut self) {
		unsafe {
			let extra_memory = self.ptr.as_ref().extra_memory;
			let layout = std::alloc::Layout::from_size_align_unchecked(
				std::mem::size_of::<KernelArgs>() + extra_memory,
				std::mem::align_of::<KernelArgs>(),
			);
			let raw_ptr = self.ptr.as_ptr().cast::<u8>();
			std::alloc::dealloc(raw_ptr, layout);
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub trait Device {
	fn name(&self) -> &str;

	fn is_cpu(&self) -> bool {
		false
	}

	fn new_buffer(self: Rc<Self>, bytes: usize) -> Result<DevicePtr, DeviceAllocError>;

	/// # Safety
	/// TODO
	unsafe fn drop_buffer(self: Rc<Self>, device_ptr: DevicePtr);

	/// # Safety
	/// TODO
	unsafe fn upload_data(
		self: Rc<Self>,
		src: NonNull<u8>,
		dst: DevicePtr,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>>;

	/// # Safety
	/// TODO
	unsafe fn download_data(
		&self,
		src: DevicePtr,
		dst: NonNull<u8>,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>>;

	/// # Safety
	/// TODO
	unsafe fn run_fragment(
		&self,
		compilation: &Compilation,
		fragment: FragmentIndex,
		args: *const KernelArgs,
	) -> Result<(), ErrPack<TensorOpError>>;
}

//--------------------------------------------------------------------------------------------------
