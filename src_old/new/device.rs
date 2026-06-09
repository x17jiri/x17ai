//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::DeviceAllocError;

pub mod cpu;

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

#[repr(C)]
pub struct KernelInput {
	pub stride_bytes: [usize; 3],
	pub buf: DevicePtr,
}

#[repr(transparent)]
pub struct KernelScalarInput {
	pub scalar: f64,
}

#[repr(transparent)]
pub struct KernelOutput {
	pub buf: DevicePtr,
}

#[repr(C)]
pub struct KernelArgs {
	shape: [usize; 3],
	extra_memory: usize,
	inp_count: usize,
	out_count: usize,
	scalar_count: usize,
	inputs: [MaybeUninit<KernelInput>; 0],
	scalars: [MaybeUninit<KernelScalarInput>; 0],
	outputs: [MaybeUninit<KernelOutput>; 0],
}

impl KernelArgs {
	/// # Panics
	/// Panics if allocation fails.
	#[allow(clippy::new_ret_no_self)]
	pub fn new(extra_memory: usize) -> KernelArgsBox {
		let input_align = std::mem::align_of::<KernelInput>();
		let scalar_align = std::mem::align_of::<KernelScalarInput>();
		let output_align = std::mem::align_of::<KernelOutput>();
		let extra_align = input_align.max(scalar_align).max(output_align);
		#[allow(clippy::panic)]
		if let struct_layout = std::alloc::Layout::new::<Self>()
			&& let Ok(extra_layout) = std::alloc::Layout::from_size_align(extra_memory, extra_align)
			&& let Ok((layout, _offset)) = struct_layout.extend(extra_layout)
			&& let Some(raw_ptr) = NonNull::new(unsafe { std::alloc::alloc(layout) })
		{
			let ptr = raw_ptr.cast();
			unsafe {
				ptr.write(Self {
					shape: [0; 3],
					extra_memory,
					inp_count: 0,
					out_count: 0,
					scalar_count: 0,
					inputs: [],
					scalars: [],
					outputs: [],
				});
			}
			KernelArgsBox { ptr }
		} else {
			cold_path();
			panic!("allocation error");
		}
	}

	pub fn set_shape(&mut self, shape: [usize; 3]) {
		self.shape = shape;
	}

	fn inputs_ptr(&self) -> *const [MaybeUninit<KernelInput>] {
		unsafe {
			let ptr = std::ptr::from_ref(self).add(1).cast();
			std::ptr::slice_from_raw_parts(ptr, self.inp_count)
		}
	}

	pub fn inputs(&self) -> &[MaybeUninit<KernelInput>] {
		unsafe { &*self.inputs_ptr() }
	}

	pub fn inputs_mut(&mut self) -> &mut [MaybeUninit<KernelInput>] {
		unsafe { &mut *self.inputs_ptr().cast_mut() }
	}

	fn outputs_ptr(&self) -> *const [MaybeUninit<KernelOutput>] {
		unsafe {
			let ptr =
				std::ptr::from_ref(self).add(1).cast::<KernelInput>().add(self.inp_count).cast();
			std::ptr::slice_from_raw_parts(ptr, self.out_count)
		}
	}

	pub fn outputs(&self) -> &[MaybeUninit<KernelOutput>] {
		unsafe { &*self.outputs_ptr() }
	}

	pub fn outputs_mut(&mut self) -> &mut [MaybeUninit<KernelOutput>] {
		unsafe { &mut *self.outputs_ptr().cast_mut() }
	}

	pub fn scalars_ptr(&self) -> *const [MaybeUninit<KernelScalarInput>] {
		unsafe {
			let ptr = std::ptr::from_ref(self)
				.add(1)
				.cast::<KernelInput>()
				.add(self.inp_count)
				.cast::<KernelOutput>()
				.add(self.out_count)
				.cast();
			std::ptr::slice_from_raw_parts(ptr, self.scalar_count)
		}
	}

	pub fn scalars(&self) -> &[MaybeUninit<KernelScalarInput>] {
		unsafe { &*self.scalars_ptr() }
	}

	pub fn scalars_mut(&mut self) -> &mut [MaybeUninit<KernelScalarInput>] {
		unsafe { &mut *self.scalars_ptr().cast_mut() }
	}

	pub fn extra_memory(inp_count: usize, out_count: usize, scalar_count: usize) -> usize {
		const {
			assert!(std::mem::align_of::<Self>() >= std::mem::align_of::<KernelInput>());
			assert!(std::mem::align_of::<KernelInput>() >= std::mem::align_of::<KernelOutput>());
			assert!(
				std::mem::align_of::<KernelOutput>() >= std::mem::align_of::<KernelScalarInput>()
			);
		}
		inp_count * std::mem::size_of::<KernelInput>()
			+ out_count * std::mem::size_of::<KernelOutput>()
			+ scalar_count * std::mem::size_of::<KernelScalarInput>()
	}

	pub fn set_counts(
		&mut self,
		inp_count: usize,
		out_count: usize,
		scalar_count: usize,
	) -> Result<(), ()> {
		if self.extra_memory >= Self::extra_memory(inp_count, out_count, scalar_count) {
			self.inp_count = inp_count;
			self.out_count = out_count;
			self.scalar_count = scalar_count;
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

impl std::ops::Deref for KernelArgsBox {
	type Target = KernelArgs;

	fn deref(&self) -> &KernelArgs {
		unsafe { self.ptr.as_ref() }
	}
}

impl std::ops::DerefMut for KernelArgsBox {
	fn deref_mut(&mut self) -> &mut KernelArgs {
		unsafe { self.ptr.as_mut() }
	}
}

//--------------------------------------------------------------------------------------------------

pub trait Device {
	fn name(&self) -> &str;

	fn is_cpu(&self) -> bool {
		false
	}

	/// # Safety
	/// - The returned buffer can only be used by this device.
	unsafe fn new_buffer(&self, bytes: usize) -> Result<DevicePtr, DeviceAllocError>;

	/// # Safety
	/// TODO
	unsafe fn drop_buffer(&self, device_ptr: DevicePtr);

	/// # Safety
	/// TODO
	unsafe fn upload_data(
		&self,
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

	/*
	/// # Safety
	/// TODO
	unsafe fn run_fragment(
		&self,
		compilation: &CompiledExpr,
		fragment: FragmentIndex,
		args: &KernelArgs,
	) -> Result<(), ErrPack<TensorOpError>>;
	*/
}

//--------------------------------------------------------------------------------------------------
