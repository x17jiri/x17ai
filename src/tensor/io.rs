//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::tensor::device::cpu::math::FromToF64;
use crate::tensor::generic::map::{DD, ND};
use crate::tensor::math::ElemWise;
use crate::tensor::{TensorOpError, generic};

use super::Tensor;

//--------------------------------------------------------------------------------------------------

pub fn write_bin(
	src: &Tensor,
	writer: &mut dyn std::io::Write,
) -> Result<(), ErrPack<TensorOpError>> {
	let executor = src.executor();
	ElemWise::new([], [src])?.run(|[], [src]| {
		executor.write_bin(src, writer)?;
		Ok(())
	})
}
/*
pub fn write_file<P: AsRef<std::path::Path>>(src: &Tensor, path: P) -> Result<()> {
	let mut file = std::fs::File::create(path)?;
	let mut header_bytes = [Default::default(); file_header::HEADER_LEN];
	file_header::write_header(&mut header_bytes, src.dtype(), &src.map.dims)?;
	file.write_all(&header_bytes)?;
	write_bin(src, &mut file)?;
	Ok(())
}
*/
//--------------------------------------------------------------------------------------------------

pub fn read_bin(
	dst: &Tensor,
	reader: &mut dyn std::io::Read,
) -> Result<(), ErrPack<TensorOpError>> {
	let executor = dst.executor();
	ElemWise::new([dst], [])?.run(|[dst], []| {
		executor.read_bin(dst, reader)?;
		Ok(())
	})
}

/*
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
*/
//--------------------------------------------------------------------------------------------------

fn fmt_0d<T: Copy>(
	f: &mut std::fmt::Formatter,
	tensor: generic::Tensor<ND<0>, &[T]>,
	mut fmt_one: impl FnMut(&mut std::fmt::Formatter, T) -> std::fmt::Result,
) -> std::fmt::Result {
	fmt_one(f, tensor[[]])?;
	Ok(())
}

fn fmt_1d<T: Copy>(
	f: &mut std::fmt::Formatter,
	tensor: generic::Tensor<ND<1>, &[T]>,
	mut fmt_one: impl FnMut(&mut std::fmt::Formatter, T) -> std::fmt::Result,
) -> std::fmt::Result {
	write!(f, "[")?;
	let mut first = true;
	#[allow(clippy::unwrap_used)]
	for elem in tensor.iter_along_axis(0).unwrap() {
		if !first {
			write!(f, ", ")?;
		}
		first = false;

		fmt_one(f, elem[[]])?;
	}
	write!(f, "]")?;
	Ok(())
}

fn fmt_Nd<T: Copy>(
	f: &mut std::fmt::Formatter,
	tensor: &generic::Tensor<&DD, &[T]>,
	indent: usize,
	fmt_one: &mut impl FnMut(&mut std::fmt::Formatter, T) -> std::fmt::Result,
) -> std::fmt::Result {
	#[allow(clippy::unwrap_used)]
	match tensor.ndim() {
		0 => {
			let tensor = tensor.conv_map_ref().unwrap();
			fmt_0d(f, tensor, fmt_one)?;
		},
		1 => {
			let tensor = tensor.conv_map_ref().unwrap();
			fmt_1d(f, tensor, fmt_one)?;
		},
		_ => {
			let indent_str = "\t".repeat(indent);
			writeln!(f, "{indent_str}[")?;
			for sub_tensor in tensor.iter_along_axis(0).unwrap() {
				write!(f, "{indent_str}\t")?;
				fmt_Nd(f, &sub_tensor.ref_map(), indent + 1, fmt_one)?;
				writeln!(f, ",")?;
			}
			write!(f, "{indent_str}]")?;
		},
	}
	Ok(())
}

fn fmt_one<T: FromToF64>(f: &mut std::fmt::Formatter, val: T) -> std::fmt::Result {
	let val = val.to_f64();
	if val >= 0.0 {
		write!(f, " ")?;
	}
	write!(f, "{val:.7}")
}

impl<T: FromToF64> std::fmt::Display for generic::Tensor<&DD, &[T]> {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Tensor(")?;
		fmt_Nd(f, self, 0, &mut fmt_one)?;
		write!(f, ")")
	}
}

//--------------------------------------------------------------------------------------------------
