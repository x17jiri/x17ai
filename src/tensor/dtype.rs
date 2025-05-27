// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::num::NonZeroU8;

pub const MAX_DTYPE_ALIGN: usize = 8; // 64-bit

pub trait HasDType {
	const dtype: DType;
}

impl HasDType for u8 {
	const dtype: DType = DType {
		kind: DTypeKind::Uint,
		bits: NonZeroU8::new(8).unwrap(),
	};
}

impl HasDType for f32 {
	const dtype: DType = DType {
		kind: DTypeKind::Float,
		bits: NonZeroU8::new(32).unwrap(),
	};
}

impl HasDType for f64 {
	const dtype: DType = DType {
		kind: DTypeKind::Float,
		bits: NonZeroU8::new(64).unwrap(),
	};
}

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

	pub fn array_bytes(&self, elems: usize) -> Option<usize> {
		debug_assert!(self.bits.is_power_of_two());
		if self.bits.get() < 8 {
			todo!("bitfields");
		}
		self.bytes().checked_mul(elems)
	}
}

#[derive(Debug, PartialEq, Clone, Copy)]
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
