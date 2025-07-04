//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::generic::map::SizeAndStride;

use super::super::dim_merger::*;

// Note: The expected tensors were generated by:
//
//    python gen_test_data.py rms_norm

#[allow(clippy::panic_in_result_fn)]
#[test]
fn test_dim_merger() -> Result<(), DimMergerError> {
	// This was a real use case that failed in the past.

	let d_lin = [
		SizeAndStride { size: 4, stride: 14 }, //
		SizeAndStride { size: 1, stride: 7 },
		SizeAndStride { size: 7, stride: 1 },
	];
	let d_gate = [
		SizeAndStride { size: 4, stride: 14 }, //
		SizeAndStride { size: 1, stride: 7 },
		SizeAndStride { size: 7, stride: 1 },
	];
	let lin = [
		// SizeAndStride { size: 1, stride: 0 },
		SizeAndStride { size: 4, stride: 14 },
		SizeAndStride { size: 7, stride: 1 },
	];
	let gate = [
		// SizeAndStride { size: 1, stride: 0 },
		SizeAndStride { size: 4, stride: 14 },
		SizeAndStride { size: 7, stride: 1 },
	];
	let out = [
		// SizeAndStride { size: 1, stride: 0 },
		SizeAndStride { size: 4, stride: 7 },
		SizeAndStride { size: 7, stride: 1 },
	];

	let dims = DimMerger::merge::<4>([&d_lin, &d_gate, &lin, &gate, &out])?;

	let expected_dims = [
		MergedDim { size: 1, strides: [1, 1, 1, 1, 1] },
		MergedDim { size: 4, strides: [14, 14, 0, 0, 0] },
		MergedDim { size: 4, strides: [0, 0, 14, 14, 7] },
		MergedDim { size: 7, strides: [1, 1, 1, 1, 1] },
	];

	assert_eq!(dims, expected_dims);
	Ok(())
}
