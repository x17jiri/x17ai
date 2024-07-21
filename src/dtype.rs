// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::fmt;

pub const MAX_DTYPE_ALIGN: usize = 8; // 64-bit

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DType {
	pub kind: DTypeKind,
	pub bits: u8,
}

impl DType {
	pub fn is_float(&self) -> bool {
		self.kind == DTypeKind::Float
	}

	pub fn bits(&self) -> usize {
		self.bits as usize
	}

	// NOTE: We don't support types with size 0.
	// However, this function will return 0 if the type uses 1, 2 or 4 bits.
	pub fn bytes(&self) -> usize {
		(self.bits as usize) / 8
	}

	pub fn array_bytes(&self, elems: usize) -> Option<usize> {
		debug_assert!(self.bits.is_power_of_two());
		if self.bits < 8 {
			todo!("bitfields");
		}
		self.bytes().checked_mul(elems)
	}

	pub fn f32() -> Self {
		Self { kind: DTypeKind::Float, bits: 32 }
	}
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
