//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::dim_merger::{DimMerger, DimMergerError};
use crate::tensor::generic;
use crate::tensor::generic::map::{DD, ND, SizeAndStride};
use crate::util::FromToF64;

use super::Tensor;

//--------------------------------------------------------------------------------------------------

// TODO - replace with something like tensor.to_nd()
pub fn merge_dims<const N: usize>(tensor: &Tensor) -> Result<ND<N>, DimMergerError> {
	let dims = DimMerger::merge::<N>([tensor.map().dims.as_slice()])?;
	Ok(ND {
		dims: std::array::from_fn(|i| SizeAndStride {
			size: dims[i].size,
			stride: dims[i].strides[0],
		}),
		offset: tensor.map().offset,
	})
}

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
