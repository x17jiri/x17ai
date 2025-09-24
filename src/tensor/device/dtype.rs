//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::num::NonZeroU8;

pub const MAX_DTYPE_ALIGN: usize = 8; // 64-bit

pub trait HasDType {
	const dtype: DType;
}

impl HasDType for u8 {
	const dtype: DType = DType {
		kind: DTypeKind::Uint,
		bits: NonZeroU8::new(8).unwrap(),
		bytes: 1,
		reserved: 0,
	};
}

impl HasDType for f32 {
	const dtype: DType = DType {
		kind: DTypeKind::Float,
		bits: NonZeroU8::new(32).unwrap(),
		bytes: 4,
		reserved: 0,
	};
}

impl HasDType for f64 {
	const dtype: DType = DType {
		kind: DTypeKind::Float,
		bits: NonZeroU8::new(64).unwrap(),
		bytes: 8,
		reserved: 0,
	};
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct DType {
	kind: DTypeKind,
	bits: NonZeroU8,
	bytes: u8,
	reserved: u8,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnknownDTypeError;

impl std::str::FromStr for DType {
	type Err = UnknownDTypeError;

	fn from_str(s: &str) -> Result<Self, UnknownDTypeError> {
		match s {
			"f32" => Ok(f32::dtype),
			"f64" => Ok(f64::dtype),
			_ => {
				cold_path();
				Err(UnknownDTypeError)
			},
		}
	}
}

impl DType {
	pub fn is_float(&self) -> bool {
		self.kind == DTypeKind::Float
	}

	pub fn bits(&self) -> usize {
		usize::from(self.bits.get())
	}

	/// NOTE: We don't support types with size 0.
	/// However, this function will return 0 if the type uses 1, 2 or 4 bits.
	pub fn bytes(&self) -> usize {
		self.bytes as usize
	}

	pub fn align(&self) -> usize {
		self.bytes().max(1)
	}

	pub fn array_bytes(&self, elems: usize) -> Option<usize> {
		debug_assert!(self.bits.is_power_of_two());
		if self.bytes < 1 {
			todo!("bitfields");
		}
		self.bytes().checked_mul(elems)
	}

	/// # Safety
	///
	/// `elems` must be such that the total size in bytes does not overflow.
	pub unsafe fn array_bytes_unchecked(&self, elems: usize) -> usize {
		debug_assert!(self.bits.is_power_of_two());
		if self.bytes < 1 {
			todo!("bitfields");
		}
		self.bytes() * elems
	}

	pub fn max(self, other: Self) -> Option<Self> {
		if self.kind != other.kind {
			cold_path();
			return None;
		}
		Some(if self.bits >= other.bits { self } else { other })
	}
}

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DTypeKind {
	Float,
	Int,
	Uint,
}

impl std::fmt::Display for DType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		let kind = match self.kind {
			DTypeKind::Float => "f",
			DTypeKind::Int => "i",
			DTypeKind::Uint => "u",
		};
		write!(f, "{}{}", kind, self.bits)
	}
}
