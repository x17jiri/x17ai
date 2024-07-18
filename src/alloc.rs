// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;

pub struct BumpAllocator {
	buffer: Rc<dyn Buffer>,
	offset: usize,
	capacity: usize,
}

impl BumpAllocator {
	pub fn new(buffer: Rc<dyn Buffer>) -> Self {
		let capacity = buffer.capacity();
		Self { buffer, offset: 0, capacity }
	}

	pub fn alloc_tensor(&mut self, dtype: DType, shape: Rc<Shape>) -> Tensor {
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
