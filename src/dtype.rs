// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::fmt;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DType {
	Float(u8),
	Int(u8),
	Uint(u8),
}

impl DType {
	pub fn bits(&self) -> usize {
		match self {
			DType::Float(b) => *b as usize,
			DType::Int(b) => *b as usize,
			DType::Uint(b) => *b as usize,
		}
	}
}

impl fmt::Display for DType {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			DType::Float(b) => write!(f, "f{}", b),
			DType::Int(b) => write!(f, "i{}", b),
			DType::Uint(b) => write!(f, "u{}", b),
		}
	}
}
