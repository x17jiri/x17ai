//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;

use crate::util::intrusive_ref_cell::{BorrowError, BorrowMutError};
use crate::{ErrExtra, ErrPack};

use super::device::DevBufAllocFailedError;
use super::device::dtype::DTypeMismatchError;
use super::dim_index::DimIndexOutOfBoundsError;
use super::map::{
	IndexOutOfBoundsError, NotEnoughDimensionsError, SelectError, TensorSizeOverflowError,
};
use super::shape::{DimsDontMatchError, ReshapeError, TooManyMergedDimensionsError};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TensorUnsafeError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnsupportedDTypeError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct NotContiguousError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OverlappingOutputError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InvalidBufferSizeError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ShapeMismatchError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorOpError {
	DimsDontMatch,
	TooManyMergedDimensions,
	CannotBorrow,
	CannotBorrowMut,
	MissingReduceDimension,
	DimIndexOutOfBounds,
	IndexOutOfBounds,
	ElementsOverflow,
	NotEnoughDimensions,
	NewBufUnsupportedDType,
	UnsupportedDType,
	DevBufAllocFailed,
	IncompatibleStridesForMerge,
	InvalidValue,
	OverlappingOutput,
	ShapeMismatch,
	DTypeMismatch,
	UnsafeTensor,
	NotContiguous,
	NotContiguousOrBroadcasted,
	InvalidShape,
	InvalidDType,
	InvalidDevice,
	InvalidBufferSize,
	InvalidReshape,
	IOError,
	DeviceError,
}

impl From<DevBufAllocFailedError> for TensorOpError {
	fn from(_: DevBufAllocFailedError) -> Self {
		Self::DevBufAllocFailed
	}
}

impl From<DevBufAllocFailedError> for ErrPack<TensorOpError> {
	fn from(_: DevBufAllocFailedError) -> Self {
		Self {
			code: TensorOpError::DevBufAllocFailed,
			extra: None,
		}
	}
}

impl From<DimsDontMatchError> for TensorOpError {
	fn from(_: DimsDontMatchError) -> Self {
		Self::DimsDontMatch
	}
}

impl From<DimsDontMatchError> for ErrPack<TensorOpError> {
	fn from(_: DimsDontMatchError) -> Self {
		Self {
			code: TensorOpError::DimsDontMatch,
			extra: None,
		}
	}
}

impl From<TooManyMergedDimensionsError> for TensorOpError {
	fn from(_: TooManyMergedDimensionsError) -> Self {
		Self::TooManyMergedDimensions
	}
}

impl From<TooManyMergedDimensionsError> for ErrPack<TensorOpError> {
	fn from(_: TooManyMergedDimensionsError) -> Self {
		Self {
			code: TensorOpError::TooManyMergedDimensions,
			extra: None,
		}
	}
}

impl From<BorrowError> for TensorOpError {
	fn from(_: BorrowError) -> Self {
		Self::CannotBorrow
	}
}

impl From<BorrowError> for ErrPack<TensorOpError> {
	fn from(_: BorrowError) -> Self {
		Self {
			code: TensorOpError::CannotBorrow,
			extra: None,
		}
	}
}

impl From<BorrowMutError> for TensorOpError {
	fn from(_: BorrowMutError) -> Self {
		Self::CannotBorrowMut
	}
}

impl From<BorrowMutError> for ErrPack<TensorOpError> {
	fn from(_: BorrowMutError) -> Self {
		debug_assert!(false, "Cannot get mutable borrow");
		Self {
			code: TensorOpError::CannotBorrowMut,
			extra: None,
		}
	}
}

impl From<NotEnoughDimensionsError> for TensorOpError {
	fn from(_: NotEnoughDimensionsError) -> Self {
		Self::NotEnoughDimensions
	}
}

impl From<NotEnoughDimensionsError> for ErrPack<TensorOpError> {
	fn from(_: NotEnoughDimensionsError) -> Self {
		Self {
			code: TensorOpError::NotEnoughDimensions,
			extra: None,
		}
	}
}

impl From<TensorSizeOverflowError> for TensorOpError {
	fn from(_: TensorSizeOverflowError) -> Self {
		Self::ElementsOverflow
	}
}

impl From<TensorSizeOverflowError> for ErrPack<TensorOpError> {
	fn from(_: TensorSizeOverflowError) -> Self {
		Self {
			code: TensorOpError::ElementsOverflow,
			extra: None,
		}
	}
}

/*
impl From<ReplaceTailError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(err: ReplaceTailError) -> Self {
		match err {
			ReplaceTailError::ElementsOverflow => Self::ElementsOverflow,
			ReplaceTailError::NotEnoughDimensions => Self::NotEnoughDimensions,
		}
	}
}

impl From<ReplaceTailError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: ReplaceTailError) -> Self {
		Self { code: err.into(), extra: None }
	}
}

impl From<SelectError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(err: SelectError) -> Self {
		match err {
			SelectError::DimIndexOutOfBounds => Self::DimIndexOutOfBounds,
			SelectError::IndexOutOfBounds => Self::IndexOutOfBounds,
		}
	}
}

impl From<SelectError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: SelectError) -> Self {
		Self { code: err.into(), extra: None }
	}
}
*/

impl From<IndexOutOfBoundsError> for ErrPack<TensorOpError> {
	fn from(_: IndexOutOfBoundsError) -> Self {
		Self {
			code: TensorOpError::IndexOutOfBounds,
			extra: None,
		}
	}
}

impl From<TensorUnsafeError> for TensorOpError {
	fn from(_: TensorUnsafeError) -> Self {
		Self::UnsafeTensor
	}
}

impl From<ErrPack<TensorUnsafeError>> for ErrPack<TensorOpError> {
	fn from(err: ErrPack<TensorUnsafeError>) -> Self {
		Self { code: err.code.into(), extra: err.extra }
	}
}

impl From<std::io::Error> for ErrPack<TensorOpError> {
	fn from(err: std::io::Error) -> Self {
		Self {
			code: TensorOpError::IOError,
			extra: Some(Box::new(ErrExtra {
				message: Cow::from("IO error occurred"),
				nested: Some(Box::new(err)),
			})),
		}
	}
}

impl From<DimIndexOutOfBoundsError> for ErrPack<TensorOpError> {
	fn from(_: DimIndexOutOfBoundsError) -> Self {
		Self {
			code: TensorOpError::DimIndexOutOfBounds,
			extra: None,
		}
	}
}

impl From<DTypeMismatchError> for ErrPack<TensorOpError> {
	fn from(_: DTypeMismatchError) -> Self {
		Self {
			code: TensorOpError::DTypeMismatch,
			extra: None,
		}
	}
}

impl From<ShapeMismatchError> for ErrPack<TensorOpError> {
	fn from(_: ShapeMismatchError) -> Self {
		Self {
			code: TensorOpError::ShapeMismatch,
			extra: None,
		}
	}
}

impl From<UnsupportedDTypeError> for ErrPack<TensorOpError> {
	fn from(_: UnsupportedDTypeError) -> Self {
		Self {
			code: TensorOpError::UnsupportedDType,
			extra: None,
		}
	}
}

impl From<NotContiguousError> for ErrPack<TensorOpError> {
	fn from(_: NotContiguousError) -> Self {
		Self {
			code: TensorOpError::NotContiguous,
			extra: None,
		}
	}
}

impl From<OverlappingOutputError> for ErrPack<TensorOpError> {
	fn from(_: OverlappingOutputError) -> Self {
		Self {
			code: TensorOpError::OverlappingOutput,
			extra: None,
		}
	}
}

impl From<InvalidBufferSizeError> for ErrPack<TensorOpError> {
	fn from(_: InvalidBufferSizeError) -> Self {
		Self {
			code: TensorOpError::InvalidBufferSize,
			extra: None,
		}
	}
}

impl From<ReshapeError> for ErrPack<TensorOpError> {
	fn from(_: ReshapeError) -> Self {
		Self {
			code: TensorOpError::InvalidReshape,
			extra: None,
		}
	}
}

impl From<SelectError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: SelectError) -> Self {
		match err {
			SelectError::DimIndexOutOfBounds => Self {
				code: TensorOpError::DimIndexOutOfBounds,
				extra: None,
			},
			SelectError::IndexOutOfBounds => Self {
				code: TensorOpError::IndexOutOfBounds,
				extra: None,
			},
		}
	}
}

impl From<TensorOpError> for ErrPack<TensorOpError> {
	fn from(err: TensorOpError) -> Self {
		Self { code: err, extra: None }
	}
}

//--------------------------------------------------------------------------------------------------
