// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::fmt;
use std::num::NonZeroU8;

use super::{TensorSize, tensor_size_to_usize};

pub const MAX_DTYPE_ALIGN: usize = 8; // 64-bit

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DType {
	pub kind: DTypeKind,
	pub bits: NonZeroU8,
}

impl DType {
	pub fn is_float(&self) -> bool {
		self.kind == DTypeKind::Float
	}

	pub fn bits(&self) -> usize {
		usize::from(self.bits.get())
	}

	// NOTE: We don't support types with size 0.
	// However, this function will return 0 if the type uses 1, 2 or 4 bits.
	pub fn bytes(&self) -> usize {
		usize::from(self.bits.get()) / 8
	}

	pub fn array_bytes(&self, elems: TensorSize) -> Option<usize> {
		debug_assert!(self.bits.is_power_of_two());
		if self.bits.get() < 8 {
			todo!("bitfields");
		}
		self.bytes().checked_mul(tensor_size_to_usize(elems))
	}

	pub const F32: Self = Self {
		kind: DTypeKind::Float,
		bits: NonZeroU8::new(32).unwrap(),
	};
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DTypeKind {
	Float,
	Int,
	Uint,
}

impl fmt::Display for DType {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let kind = match self.kind {
			DTypeKind::Float => "f",
			DTypeKind::Int => "i",
			DTypeKind::Uint => "u",
		};
		write!(f, "{}{}", kind, self.bits)
	}
}
