//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use crate::tensor::HasDType;
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::cpu::cpu_float_methods::FromToF64;
use crate::tensor::dim_merger::{DimMerger, DimMergerError};
use crate::tensor::generic::map::{DD, Map, ND, SizeAndStride};

use super::Tensor;

//--------------------------------------------------------------------------------------------------

pub fn merge_dims<const N: usize>(tensor: &Tensor) -> Result<ND<N>, DimMergerError> {
	let dims = DimMerger::merge::<N>([tensor.map().dims.as_slice()])?;
	#[allow(clippy::indexing_slicing)]
	Ok(ND {
		dims: std::array::from_fn(|i| SizeAndStride {
			size: dims[i].size,
			stride: dims[i].strides[0],
		}),
		offset: tensor.map().offset,
	})
}

//--------------------------------------------------------------------------------------------------

#[allow(clippy::indexing_slicing)]
fn fmt_0d<T: Copy>(
	f: &mut std::fmt::Formatter,
	(buf, offset): (&[T], usize),
	mut fmt_one: impl FnMut(&mut std::fmt::Formatter, T) -> std::fmt::Result,
) -> std::fmt::Result {
	fmt_one(f, buf[offset])?;
	Ok(())
}

#[allow(clippy::indexing_slicing)]
fn fmt_1d<T: Copy>(
	f: &mut std::fmt::Formatter,
	(buf, offset, dim): (&[T], usize, SizeAndStride),
	indent: usize,
	mut fmt_one: impl FnMut(&mut std::fmt::Formatter, T) -> std::fmt::Result,
) -> std::fmt::Result {
	write!(f, "[")?;
	for i in 0..dim.size {
		if i != 0 {
			write!(f, ", ")?;
		}
		if dim.size > 4 && i % 4 == 0 {
			writeln!(f)?;
			for _ in 0..indent + 1 {
				write!(f, "\t")?;
			}
		}
		fmt_one(f, buf[offset + i * dim.stride])?;
	}
	if dim.size > 4 {
		writeln!(f)?;
		for _ in 0..indent {
			write!(f, "\t")?;
		}
	}
	write!(f, "]")?;
	Ok(())
}

#[allow(clippy::indexing_slicing)]
fn fmt_Nd<T: Copy>(
	f: &mut std::fmt::Formatter,
	(buf, dd_map): (&[T], &DD),
	dim_index: usize,
	offset: usize,
	indent: usize,
	fmt_one: &mut impl FnMut(&mut std::fmt::Formatter, T) -> std::fmt::Result,
) -> std::fmt::Result {
	let ndim = dd_map.ndim();
	if dim_index >= ndim {
		fmt_0d(f, (buf, offset), fmt_one)?;
	} else if dim_index == ndim - 1 {
		let dims = dd_map.dims.as_slice();
		let dim = dims[dim_index];
		fmt_1d(f, (buf, offset, dim), indent, fmt_one)?;
	} else {
		let dims = dd_map.dims.as_slice();
		let dim = dims[dim_index];
		for _ in 0..indent {
			write!(f, "\t")?;
		}
		writeln!(f, "[")?;
		for i in 0..dim.size {
			for _ in 0..indent + 1 {
				write!(f, "\t")?;
			}
			fmt_Nd(f, (buf, dd_map), dim_index + 1, offset + i * dim.stride, indent + 1, fmt_one)?;
			writeln!(f, ",")?;
		}
		for _ in 0..indent {
			write!(f, "\t")?;
		}
		write!(f, "]")?;
	}
	Ok(())
}

fn fmt_one<T: FromToF64>(f: &mut std::fmt::Formatter, val: T) -> std::fmt::Result {
	let val = val.to_f64();
	let alignment = if val < 0.0 { "" } else { " " };
	write!(f, "{alignment}{val:.7e}")
}

pub fn fmt_tensor<T: FromToF64>(
	f: &mut std::fmt::Formatter,
	map: &DD,
	buf: &[T],
) -> std::fmt::Result {
	write!(f, "Tensor(")?;
	fmt_Nd(f, (buf, map), 0, map.offset, 0, &mut fmt_one)?;
	write!(f, ")")
}

impl std::fmt::Display for Tensor {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		let map = self.map();
		let buf = self.buf();
		let Ok(buf) = buf.try_borrow() else {
			cold_path();
			write!(f, "Tensor(<cannot borrow>)")?;
			return Err(std::fmt::Error);
		};
		let dtype = buf.dtype();
		#[allow(clippy::single_match_else)]
		match dtype {
			f32::dtype => {
				let Ok(slice) = CPUDevice::buf_as_slice::<f32>(&buf) else {
					// TODO - could move to CPU
					cold_path();
					write!(f, "Tensor(<tensor is not on CPU>)")?;
					return Err(std::fmt::Error);
				};
				fmt_tensor(f, map, slice)
			},
			_ => {
				cold_path();
				write!(f, "Tensor(<unsupported dtype {dtype}>)")?;
				Err(std::fmt::Error)
			},
		}
	}
}

//--------------------------------------------------------------------------------------------------
