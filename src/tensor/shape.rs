//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use super::DType;
use super::map::{ElementsOverflowError, Map};

//--------------------------------------------------------------------------------------------------

pub trait Shape {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), ElementsOverflowError>;
}

impl Shape for &Map {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), ElementsOverflowError> {
		Ok(self.new_like(dtype))
	}
}

impl Shape for &[usize] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), ElementsOverflowError> {
		Map::new(self, dtype)
	}
}

impl<const N: usize> Shape for &[usize; N] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), ElementsOverflowError> {
		Map::new(self, dtype)
	}
}

impl Shape for &mut [usize] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), ElementsOverflowError> {
		Map::new(self, dtype)
	}
}

impl<const N: usize> Shape for &mut [usize; N] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), ElementsOverflowError> {
		Map::new(self, dtype)
	}
}

//--------------------------------------------------------------------------------------------------
