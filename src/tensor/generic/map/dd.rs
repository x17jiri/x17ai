//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::intrinsics::cold_path;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use crate::tensor::generic::dim_index::DimIndexOutOfBoundsError;
use crate::tensor::generic::map::{
	ElementsOverflowError, IncompatibleStridesError, MergeAllDims, MergeDims, MergeDimsError,
	NotEnoughDimensionsError, ReshapeLastDim, ReshapeLastDimError, Select, SelectError,
	StrideCounter, StrideCounterUnchecked, merge_dims,
};

use super::{Map, SizeAndStride, Transpose};

//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------

