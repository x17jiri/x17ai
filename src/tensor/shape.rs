//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use super::DType;
use super::map::{Map, TensorSizeOverflowError};

//--------------------------------------------------------------------------------------------------

pub trait Shape {
	/// Converts the shape to a `Map` with the specified `dtype`.
	///
	/// The new map is contiguous and has the same number of elements as specified by the shape.
	///
	/// Returns the new map and required buffer size in bytes.
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError>;
}

impl Shape for &Map {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		self.new_like(dtype)
	}
}

impl Shape for &[usize] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		Map::new(self, dtype)
	}
}

impl<const N: usize> Shape for &[usize; N] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		Map::new(self, dtype)
	}
}

impl Shape for &mut [usize] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		Map::new(self, dtype)
	}
}

impl<const N: usize> Shape for &mut [usize; N] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		Map::new(self, dtype)
	}
}

//--------------------------------------------------------------------------------------------------
