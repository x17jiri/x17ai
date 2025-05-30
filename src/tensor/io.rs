// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::intrinsics::cold_path;
use std::io::{Read, Write};

use super::math::__elem_wise;
use super::{HasDType, Tensor};

//--------------------------------------------------------------------------------------------------

pub fn write_bin(src: &Tensor, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
	let executor = src.buffer.executor();
	let mut result = Ok(());
	__elem_wise([src], |[src]| {
		if result.is_ok() {
			result = executor.write_bin(&src, writer);
		}
	});
	result
}

pub fn write_file<P: AsRef<std::path::Path>>(src: &Tensor, path: P) -> std::io::Result<()> {
	let mut file = std::fs::File::create(path)?;
	let mut header_bytes = [Default::default(); file_header::HEADER_LEN];
	file_header::write_header(&mut header_bytes, src.dtype, &src.dims)?;
	file.write_all(&header_bytes)?;
	write_bin(src, &mut file)?;
	Ok(())
}

//--------------------------------------------------------------------------------------------------

pub fn read_bin(dst: &Tensor, reader: &mut dyn std::io::Read) -> std::io::Result<()> {
	let executor = dst.buffer.executor();
	let mut result = Ok(());
	__elem_wise([dst], |[dst]| {
		if result.is_ok() {
			result = executor.read_bin(&dst, reader);
		}
	});
	result
}

pub fn read_file<P: AsRef<std::path::Path>>(dst: &Tensor, path: P) -> std::io::Result<()> {
	let elems = dst.elems();
	let data_len = dst.dtype.array_bytes(elems).unwrap();
	let mut file = std::fs::File::open(path)?;
	let mut header_bytes = [Default::default(); file_header::HEADER_LEN];
	file.read_exact(&mut header_bytes)?;
	let header = file_header::parse(header_bytes).map_err(|err| {
		cold_path();
		err
	})?;
	if header.dtype != dst.dtype {
		cold_path();
		return Err(std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			format!("Expected dtype '{}' but got '{}'", dst.dtype, header.dtype),
		));
	}
	if dst.dims.len() != header.shape.len() {
		cold_path();
		return Err(std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			format!("Expected {} dimensions but got {}", dst.ndim(), header.shape.len()),
		));
	}
	for (i, (&dim, &expected)) in header.shape.iter().zip(dst.dims.iter()).enumerate() {
		if dim.size != expected.size {
			cold_path();
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				format!("Dimension {}: expected {}, got {}", i, expected.size, dim.size),
			));
		}
	}
	// We could check the file size upfront, but doing it after the previous checks gives
	// better error messages.
	if file.metadata()?.len() != header_bytes.len() as u64 + data_len as u64 {
		cold_path();
		return Err(std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			format!(
				"Expected {} bytes of data, but got {} bytes",
				data_len,
				file.metadata()?.len() - header_bytes.len() as u64
			),
		));
	}
	read_bin(dst, &mut file)
}

//--------------------------------------------------------------------------------------------------

pub mod file_header {
	use std::intrinsics::cold_path;

	use crate::tensor::dim_vec::DimVec;
	use crate::tensor::{DType, SizeAndStride};

	pub struct ParsedHeader {
		pub dtype: DType,
		pub shape: DimVec,
	}

	pub fn parse_dtype(dtype_str: &str) -> std::io::Result<DType> {
		let Some(dtype) = DType::from_str(dtype_str) else {
			cold_path();
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				format!("Unknown dtype '{}'", dtype_str),
			));
		};
		Ok(dtype)
	}

	pub fn parse_shape(shape_str: &str) -> std::io::Result<DimVec> {
		let mut shape = DimVec::new();
		if shape_str != "," {
			for dim_str in shape_str.split(',') {
				match dim_str.parse::<usize>() {
					Ok(dim) => {
						shape.push(SizeAndStride { size: dim, stride: 0 });
					},
					Err(err) => {
						cold_path();
						return Err(match err.kind() {
							std::num::IntErrorKind::PosOverflow => std::io::Error::new(
								std::io::ErrorKind::InvalidData,
								format!(
									"Dimension size {} is too large to load into host integer type",
									dim_str
								),
							),
							_ => std::io::Error::new(
								std::io::ErrorKind::InvalidData,
								format!("Invalid dimension size '{}': {}", dim_str, err),
							),
						});
					},
				}
			}
		}
		Ok(shape)
	}

	pub const HEADER_LEN: usize = 128;
	pub const HEADER_PREFIX: &[u8] = b"\x93x17ai.Tensor--\n";
	pub const HEADER_SUFFIX: &[u8] = b"\n->\n";

	#[inline(never)]
	pub fn parse(header_bytes: [u8; HEADER_LEN]) -> std::io::Result<ParsedHeader> {
		let Some(content_bytes) = header_bytes
			.strip_prefix(HEADER_PREFIX)
			.and_then(|content| content.strip_suffix(HEADER_SUFFIX))
		else {
			cold_path();
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				"This doesn't seem to be x17ai Tensor file",
			));
		};

		let Ok(content) = std::str::from_utf8(content_bytes) else {
			cold_path();
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				"Invalid header content",
			));
		};
		let mut lines = content.split('\n');

		let Some(dtype_str) = lines.next().and_then(|line| line.strip_prefix("dtype=")) else {
			cold_path();
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				"Cannot find dtype in header",
			));
		};
		let dtype = parse_dtype(dtype_str)?;

		let Some(shape_str) = lines.next().and_then(|line| line.strip_prefix("shape=")) else {
			cold_path();
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				"Cannot find shape in header",
			));
		};
		let shape = parse_shape(shape_str)?;

		for line in lines {
			if !line.is_empty() && !line.starts_with("#") {
				cold_path();
				return Err(std::io::Error::new(
					std::io::ErrorKind::InvalidData,
					format!("Unexpected line in header: '{}'", line),
				));
			}
		}

		Ok(ParsedHeader { dtype, shape })
	}

	pub fn write_header_content(
		buf: &mut [u8], dtype: DType, dims: &DimVec,
	) -> std::io::Result<()> {
		use std::io::Write;
		let mut content = std::io::Cursor::new(buf);
		write!(content, "dtype={}\nshape=", dtype)?;
		if dims.is_empty() {
			write!(content, ",")?;
		} else {
			for (i, &dim) in dims.iter().enumerate() {
				if i > 0 {
					write!(content, ",")?;
				}
				write!(content, "{}", dim.size)?;
			}
		}
		let (_, rest) = content.split_mut();
		if !rest.is_empty() {
			rest[0] = b'\n';
			rest[1..].fill(b'#');
		}
		Ok(())
	}

	pub fn write_header(
		header_bytes: &mut [u8; HEADER_LEN], dtype: DType, dims: &DimVec,
	) -> std::io::Result<()> {
		let prefix_end = HEADER_PREFIX.len();
		let suffix_start = HEADER_LEN - HEADER_SUFFIX.len();
		header_bytes[..prefix_end].copy_from_slice(HEADER_PREFIX);
		header_bytes[suffix_start..].copy_from_slice(HEADER_SUFFIX);
		write_header_content(&mut header_bytes[prefix_end..suffix_start], dtype, dims)
	}
}

//--------------------------------------------------------------------------------------------------

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
		#[cfg(target_endian = "big")]
		{
			todo!("Device::load_from_reader() expects little-endian data");
		}
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
		#[cfg(target_endian = "big")]
		{
			todo!("Device::load_from_reader() expects little-endian data");
		}
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
		#[cfg(target_endian = "big")]
		{
			todo!("Device::load_from_reader() expects little-endian data");
		}
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
