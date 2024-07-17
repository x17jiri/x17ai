// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::boxed::Box;
use std::cell::{Cell, RefCell};
use std::fmt;

pub struct CPUDevice {
	name: String,
	rng: RefCell<Rng>,
}

impl CPUDevice {
	pub fn new(name: String) -> Rc<Self> {
		Rc::new(Self {
			name,
			rng: RefCell::new(Rng::new_default()),
		})
	}
}

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn new_buffer(self: Rc<Self>, size_bytes: usize, name: String) -> Rc<dyn Buffer> {
		Rc::new(CPUBuffer {
			name,
			device: self.clone(),
			memory: vec![Cell::new(0); size_bytes].into_boxed_slice(),
		})
	}
}

struct BatchTraversal<'a, T> {
	ptr: *const Cell<T>,
	batch_size: usize,
	input_size: usize,
	phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T> Iterator for BatchTraversal<'a, T> {
	type Item = &'a [Cell<T>];

	fn next(&mut self) -> Option<Self::Item> {
		if self.batch_size == 0 {
			return None;
		}

		let data = self.ptr;
		let len = self.input_size;
		self.ptr = unsafe { self.ptr.add(len) };

		self.batch_size -= 1;

		Some(unsafe { std::slice::from_raw_parts(data, len) })
	}
}

pub struct CPUBuffer {
	name: String,
	device: Rc<CPUDevice>,
	memory: Box<[Cell<u8>]>,
}

impl CPUBuffer {
	fn cast<T>(&self, byte_offset: usize, elems: usize) -> &[Cell<T>] {
		let ptr = self.memory.as_ptr().wrapping_add(byte_offset);
		let ptr = ptr as *const Cell<T>;
		unsafe { std::slice::from_raw_parts(ptr, elems) }
	}

	fn traversal<T>(
		&self,
		byte_offset: usize,
		batch_size: usize,
		input_size: usize,
	) -> BatchTraversal<T> {
		let ptr = self.memory.as_ptr().wrapping_add(byte_offset);
		let ptr = ptr as *const Cell<T>;
		BatchTraversal {
			ptr,
			batch_size,
			input_size,
			phantom: std::marker::PhantomData,
		}
	}

	fn zeros_f32_(&self, byte_offset: usize, elems: usize) {
		let data = self.cast::<f32>(byte_offset, elems);
		for val in data {
			val.set(0.0);
		}
	}

	fn randn_f32_(&self, byte_offset: usize, elems: usize) {
		let data = self.cast::<f32>(byte_offset, elems);
		let mut rng = self.device.rng.borrow_mut();
		for val in data {
			val.set(rng.get_normal() as f32);
		}
	}

	fn rms_norm_f32_(
		&self,
		i_byte_offset: usize,
		o_byte_offset: usize,
		batch_size: usize,
		input_size: usize,
	) {
		let i_batch = self.traversal::<f32>(i_byte_offset, batch_size, input_size);
		let o_batch = self.traversal::<f32>(o_byte_offset, batch_size, input_size);
		for (i_vec, o_vec) in i_batch.zip(o_batch) {
			let sqr_sum: f32 = i_vec.iter().map(|x| x.get() * x.get()).sum();
			let scale = ((input_size as f32) / sqr_sum).sqrt();
			for (i, o) in i_vec.iter().zip(o_vec) {
				o.set(i.get() * scale);
			}
		}
	}
	/*
		fn mm_f32(&self, a: &Tensor, b: &Tensor, c: &Tensor, prep: PrepMM) {
			if prep.a_transpose || prep.b_transpose {
				unimplemented!("TODO");
			}

			let a = self.traversal::<f32>(a.byte_offset, prep.batch_size, prep.a_rows * prep.a_cols);
			let b = self.traversal::<f32>(b.byte_offset, prep.batch_size, prep.a_cols * prep.b_cols);
			let c = self.traversal::<f32>(c.byte_offset, prep.batch_size, prep.a_rows * prep.b_cols);

			for ((a, b), c) in a.zip(b).zip(c) {
				for i in 0..prep.a_rows {
					for j in 0..prep.b_cols {
						let mut sum = 0.0;
						for k in 0..prep.a_cols {
							sum += a[i * prep.a_cols + k].get() * b[k * prep.b_cols + j].get();
						}
						c[i * prep.b_cols + j].set(sum);
					}
				}
			}
		}
	*/
	fn format_f32(&self, byte_offset: usize, f: &mut fmt::Formatter, count: usize) -> fmt::Result {
		let data = self.cast::<f32>(byte_offset, count);
		for (i, val) in data.iter().enumerate() {
			if i != 0 {
				write!(f, ", ")?;
			}
			write!(f, "{:.6}", val.get())?;
		}
		Ok(())
	}
}

impl Buffer for CPUBuffer {
	fn owns(&self, tensor: &Tensor) -> bool {
		let buf = tensor.buffer.as_ref() as *const dyn Buffer as *const u8;
		let slf = self as *const dyn Buffer as *const u8;
		buf == slf
	}

	fn zeros_(&self, tensor: &Tensor) {
		debug_assert!(self.owns(tensor));

		match tensor.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.zeros_f32_(tensor.byte_offset, tensor.shape.elems())
			},
			_ => unimplemented!(),
		}
	}

	fn randn_(&self, tensor: &Tensor) {
		debug_assert!(self.owns(tensor));

		match tensor.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.randn_f32_(tensor.byte_offset, tensor.shape.elems())
			},
			_ => unimplemented!(),
		}
	}

	fn rms_norm(&self, input: &Tensor, output: &Tensor, params: &ReduceParams) {
		debug_assert!(self.owns(input));
		debug_assert!(self.owns(output));
		debug_assert!(input.shape == output.shape);
		debug_assert!(input.dtype == output.dtype);
		let (batch_dims, input_dim) = input.shape.split(-1);
		debug_assert!(params.batch_size == batch_dims.iter().product());
		debug_assert!(params.input_size == input_dim.iter().product());

		match input.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => self.rms_norm_f32_(
				input.byte_offset,
				output.byte_offset,
				params.batch_size,
				params.input_size,
			),
			_ => unimplemented!(),
		}
	}
	/*
		fn mm(&self, a: &Tensor, b: &Tensor, c: &Tensor) {
			// TODO - could relax this to just check that the tensors are on the same device
			debug_assert!(self.owns(a) && self.owns(b) && self.owns(c));

			let prep = prep_mm(a, b, c);

			match prep.dtype {
				DType { kind: DTypeKind::Float, bits: 32 } => self.mm_f32(a, b, c, prep),
				_ => unimplemented!(),
			}
		}
	*/
	fn format(
		&self,
		byte_offset: usize,
		dtype: DType,
		f: &mut fmt::Formatter,
		count: usize,
	) -> fmt::Result {
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => self.format_f32(byte_offset, f, count),
			_ => unimplemented!(),
		}
	}
}
