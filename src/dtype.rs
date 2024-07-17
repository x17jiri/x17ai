// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::fmt;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DType {
	pub kind: DTypeKind,
	pub bits: u8,
}

impl DType {
	pub fn is_float(&self) -> bool {
		self.kind == DTypeKind::Float
	}

	pub fn bytes(&self) -> usize {
		(self.bits as usize) / 8
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
