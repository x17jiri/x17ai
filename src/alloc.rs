// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;

pub trait Allocator {
	fn new_tensor(&mut self, shape: Rc<Shape>, dtype: DType) -> Tensor;
}

pub struct BumpAllocator {
	pub buffer: Rc<dyn Buffer>,
	pub offset: usize,
	pub capacity: usize,
}

impl BumpAllocator {
	pub fn new(buffer: Rc<dyn Buffer>) -> Self {
		let capacity = buf_to_base(buffer.as_ref()).capacity;
		Self { buffer, offset: 0, capacity }
	}
}

impl Allocator for BumpAllocator {
	fn new_tensor(&mut self, shape: Rc<Shape>, dtype: DType) -> Tensor {
		const MAX_BYTES: usize = (isize::MAX as usize) & !(MAX_DTYPE_ALIGN - 1);
		let bytes = match dtype.array_bytes(shape.elems()) {
			Some(b) if b <= MAX_BYTES => b,
			_ => panic!("tensor too large"),
		};

		// round bytes up to a multiple of MAX_ALIGN
		let bytes = (bytes + (MAX_DTYPE_ALIGN - 1)) & !(MAX_DTYPE_ALIGN - 1);

		if self.offset + bytes > self.capacity {
			panic!("out of memory");
		}

		let byte_offset = self.offset;
		self.offset += bytes;
		Tensor {
			shape,
			dtype,
			buffer: self.buffer.clone(),
			byte_offset,
		}
	}
}

pub struct ScopedTensor<'a> {
	pub tensor: Tensor,
	pub byte_size: usize,
	pub allocator: &'a ScopedAllocator,
}

impl<'a> ScopedTensor<'a> {
	pub fn get(&self) -> &Tensor {
		&self.tensor
	}
}

impl<'a> Drop for ScopedTensor<'a> {
	fn drop(&mut self) {
		self.allocator.bump_allocator.offset -= self.byte_size;
		assert!(self.allocator.bump_allocator.offset == self.tensor.byte_offset);
	}
}

pub struct ScopedAllocator {
	pub bump_allocator: BumpAllocator,
}

impl ScopedAllocator {
	pub fn new(buffer: Rc<dyn Buffer>) -> Self {
		Self {
			bump_allocator: BumpAllocator::new(buffer),
		}
	}

	pub fn new_tensor(&self, shape: Rc<Shape>, dtype: DType) -> ScopedTensor {
		let offset = self.bump_allocator.offset;
		let tensor = self.bump_allocator.new_tensor(shape, dtype);
		let byte_size = self.bump_allocator.offset - offset;
		ScopedTensor { tensor, byte_size, allocator: self }
	}
}
