// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use super::math::__elem_wise;
use super::{HasDType, Tensor};

//--------------------------------------------------------------------------------------------------

pub fn fill_from_reader(dst: &Tensor, reader: &mut dyn std::io::Read) -> std::io::Result<()> {
	let executor = dst.buffer.executor();
	let mut result = Ok(());
	__elem_wise([dst], |[dst]| {
		if result.is_ok() {
			result = executor.load_from_reader(&dst, reader);
		}
	});
	result
}

pub fn fill_from_file(dst: &Tensor, path: &str) -> std::io::Result<()> {
	let mut file = std::fs::File::open(path)?;
	fill_from_reader(dst, &mut file)
}

fn fmt_0d(tensor: &Tensor, f: &mut std::fmt::Formatter, offset: usize) -> std::fmt::Result {
	let executor = tensor.buffer.executor();
	let offset = tensor.offset + offset;
	let len = 1;
	let stride = 1;
	executor.format(f, tensor.buffer.as_ref(), tensor.dtype, offset, len, stride)
}

fn fmt_1d(tensor: &Tensor, f: &mut std::fmt::Formatter, offset: usize) -> std::fmt::Result {
	let executor = tensor.buffer.executor();
	let dim = tensor.dims[tensor.ndim() - 1];
	let offset = tensor.offset + offset;
	let len = dim.size;
	let stride = dim.stride;
	write!(f, "[")?;
	executor.format(f, tensor.buffer.as_ref(), tensor.dtype, offset, len, stride)?;
	write!(f, "]")
}

fn fmt_Nd(
	tensor: &Tensor, f: &mut std::fmt::Formatter, offset: usize, d: usize,
) -> std::fmt::Result {
	let indent = "\t".repeat(d);
	writeln!(f, "{indent}[")?;
	let dim = tensor.dims[d];
	for i in 0..dim.size {
		write!(f, "{indent}\t")?;
		let offset = offset + i * dim.stride;
		if d + 1 < tensor.ndim() - 1 {
			fmt_Nd(tensor, f, offset, d + 1)?;
		} else {
			fmt_1d(tensor, f, offset)?;
		}
		writeln!(f, ",")?;
	}
	write!(f, "{indent}]")
}

impl std::fmt::Display for Tensor {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Tensor(")?;
		match self.ndim() {
			0 => fmt_0d(self, f, 0)?,
			1 => fmt_1d(self, f, 0)?,
			_ => fmt_Nd(self, f, 0, 0)?,
		};
		write!(f, ")")
	}
}

//--------------------------------------------------------------------------------------------------

// The following macros are meant for debugging.
// They should be easy to use, but the current implementation is not really efficient.

pub struct DebugData1D<T: HasDType> {
	pub data: Vec<T>,
}

impl<T: HasDType> DebugData1D<T> {
	pub fn into_read(self) -> DebugData1DRead<T> {
		DebugData1DRead { data: self.data, x: 0 }
	}

	pub fn shape(&self) -> (usize,) {
		let x = self.data.len();
		(x,)
	}
}

pub struct DebugData1DRead<T: HasDType> {
	data: Vec<T>,
	x: usize,
}

impl<T: HasDType> std::io::Read for DebugData1DRead<T> {
	fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
		let vec = self.data.as_slice();
		let vec_ptr = vec.as_ptr() as *const u8;
		let vec_len = vec.len() * std::mem::size_of::<T>();
		let vec = unsafe { std::slice::from_raw_parts(vec_ptr, vec_len) };
		let count = buf.len().min(vec.len() - self.x);
		buf[..count].copy_from_slice(&vec[self.x..self.x + count]);
		self.x += count;
		Ok(count)
	}
}

pub struct DebugData2D<T: HasDType> {
	pub data: Vec<Vec<T>>,
}

impl<T: HasDType> DebugData2D<T> {
	pub fn into_read(self) -> DebugData2DRead<T> {
		DebugData2DRead { data: self.data, x: 0, y: 0 }
	}

	pub fn shape(&self) -> (usize, usize) {
		let y = self.data.len();
		let x = self.data.get(0).map_or(0, |row| row.len());
		assert!(self.data.iter().all(|row| row.len() == x), "rows have different lengths");
		(y, x)
	}
}

pub struct DebugData2DRead<T: HasDType> {
	data: Vec<Vec<T>>,
	x: usize,
	y: usize,
}

impl<T: HasDType> std::io::Read for DebugData2DRead<T> {
	fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
		let mut count = 0;
		while count == 0 && self.y < self.data.len() {
			let vec = self.data[self.y].as_slice();
			let vec_ptr = vec.as_ptr() as *const u8;
			let vec_len = vec.len() * std::mem::size_of::<T>();
			let vec = unsafe { std::slice::from_raw_parts(vec_ptr, vec_len) };
			count = buf.len().min(vec.len() - self.x);
			if count == 0 {
				self.y += 1;
				self.x = 0;
				continue;
			}

			buf[..count].copy_from_slice(&vec[self.x..self.x + count]);
			self.x += count;
		}
		Ok(count)
	}
}

pub struct DebugData3D<T: HasDType> {
	pub data: Vec<Vec<Vec<T>>>,
}

impl<T: HasDType> DebugData3D<T> {
	pub fn into_read(self) -> DebugData3DRead<T> {
		DebugData3DRead { data: self.data, x: 0, y: 0, z: 0 }
	}

	pub fn shape(&self) -> (usize, usize, usize) {
		let z = self.data.len();
		let y = self.data.get(0).map_or(0, |mat| mat.len());
		assert!(
			self.data.iter().all(|mat| mat.len() == y),
			"matrices have different numbers of rows"
		);
		let x = self.data.get(0).and_then(|mat| mat.get(0)).map_or(0, |row| row.len());
		assert!(
			self.data.iter().all(|mat| mat.iter().all(|row| row.len() == x)),
			"rows have different lengths"
		);
		(z, y, x)
	}
}

pub struct DebugData3DRead<T: HasDType> {
	data: Vec<Vec<Vec<T>>>,
	x: usize,
	y: usize,
	z: usize,
}

impl<T: HasDType> std::io::Read for DebugData3DRead<T> {
	fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
		let mut count = 0;
		while count == 0 && self.z < self.data.len() {
			let vec = self.data[self.z][self.y].as_slice();
			let vec_ptr = vec.as_ptr() as *const u8;
			let vec_len = vec.len() * std::mem::size_of::<T>();
			let vec = unsafe { std::slice::from_raw_parts(vec_ptr, vec_len) };
			count = buf.len().min(vec.len() - self.x);
			if count == 0 {
				self.y += 1;
				if self.y >= self.data[self.z].len() {
					self.y = 0;
					self.z += 1;
				}
				self.x = 0;
				continue;
			}

			buf[..count].copy_from_slice(&vec[self.x..self.x + count]);
			self.x += count;
		}
		Ok(count)
	}
}

#[macro_export]
macro_rules! debug_1d {
    ( $dt:ty; $( $x:expr ),* $(,)? ) => {
        $crate::tensor::io::DebugData1D::<$dt> {
			data: vec![$($x),*]
		}
    };
}

#[macro_export]
macro_rules! debug_2d {
    ( $dt:ty; $( [ $( $x:expr ),* ] ),* $(,)? ) => {
		$crate::tensor::io::DebugData2D::<$dt> {
        	data: vec![
				$(vec![$($x),*]),*
			]
		}
    };
}

#[macro_export]
macro_rules! debug_3d {
	( $dt:ty; $( [ $( [ $( $x:expr ),* ] ),* $(,)? ] ),* $(,)? ) => {
		$crate::tensor::io::DebugData3D::<$dt> {
			data: vec![
				$(
					vec![
						$(vec![$($x),*]),*
					]
				),*
			]
		}
	};
}

//--------------------------------------------------------------------------------------------------
