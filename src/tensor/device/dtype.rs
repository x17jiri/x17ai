//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::num::{NonZeroU32, NonZeroUsize};

//--------------------------------------------------------------------------------------------------

pub const MAX_DTYPE_ALIGN: usize = 8; // 64-bit

pub trait HasDType {
	const dtype: DType;
}

impl HasDType for u8 {
	const dtype: DType = DType::from_struct(DTypeStruct {
		kind: DTypeKind::Uint,
		shift: 3,
		is_fractional: 0,
		id: DTypeId::U8,
	});
}

impl HasDType for f32 {
	const dtype: DType = DType::from_struct(DTypeStruct {
		kind: DTypeKind::Float,
		shift: 5,
		is_fractional: 0,
		id: DTypeId::F32,
	});
}

impl HasDType for f64 {
	const dtype: DType = DType::from_struct(DTypeStruct {
		kind: DTypeKind::Float,
		shift: 6,
		is_fractional: 0,
		id: DTypeId::F64,
	});
}

#[derive(Clone, Copy)]
pub struct DTypeStruct {
	kind: DTypeKind,

	/// Shifting left by this value will convert number of elements to number of bits
	shift: u8,

	is_fractional: u8,

	id: DTypeId,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[repr(transparent)]
pub struct DType(NonZeroU32);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DTypeMismatchError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnknownDTypeError;

impl std::str::FromStr for DType {
	type Err = UnknownDTypeError;

	fn from_str(s: &str) -> Result<Self, UnknownDTypeError> {
		match s {
			"f32" => Ok(f32::dtype),
			"f64" => Ok(f64::dtype),
			_ => {
				cold_path();
				Err(UnknownDTypeError)
			},
		}
	}
}

impl DTypeStruct {
	pub const fn from_dtype(dtype: DType) -> Self {
		let bytes = dtype.0.get().to_le_bytes();
		Self {
			kind: unsafe { std::mem::transmute(bytes[0]) },
			shift: bytes[1],
			is_fractional: bytes[2],
			id: unsafe { std::mem::transmute(bytes[3]) },
		}
	}
}

impl DType {
	pub const fn from_struct(data: DTypeStruct) -> Self {
		let val =
			u32::from_le_bytes([data.kind as u8, data.shift, data.is_fractional, data.id as u8]);
		// SAFETY: DTypeId starts at 1, so val is never 0
		Self(unsafe { NonZeroU32::new_unchecked(val) })
	}

	pub fn is_float(self) -> bool {
		DTypeStruct::from_dtype(self).kind == DTypeKind::Float
	}

	pub fn kind(self) -> DTypeKind {
		DTypeStruct::from_dtype(self).kind
	}

	pub fn id(self) -> DTypeId {
		DTypeStruct::from_dtype(self).id
	}

	pub fn bits(self) -> usize {
		let data = DTypeStruct::from_dtype(self);
		1 << data.shift
	}

	pub fn floor_bytes(self) -> usize {
		self.bits() / 8
	}

	pub fn ceil_bytes(self) -> usize {
		(self.bits() + 7) / 8
	}

	pub fn exact_bytes(self) -> Option<NonZeroUsize> {
		NonZeroUsize::new(self.floor_bytes())
	}

	pub fn is_fractional(self) -> usize {
		let data = DTypeStruct::from_dtype(self);
		data.is_fractional.into()
	}

	pub fn array_bits(self, elems: usize) -> Option<usize> {
		let data = DTypeStruct::from_dtype(self);
		elems.checked_shl(data.shift.into())
	}

	/// # Safety
	///
	/// `elems` must be such that the total size in bytes does not overflow.
	pub unsafe fn array_bits_unchecked(self, elems: usize) -> usize {
		let data = DTypeStruct::from_dtype(self);
		elems << data.shift
	}
}

pub fn common_dtype(a: DType, b: DType) -> Result<DType, DTypeMismatchError> {
	let a_data = DTypeStruct::from_dtype(a);
	let b_data = DTypeStruct::from_dtype(b);
	if a_data.kind != b_data.kind {
		cold_path();
		return Err(DTypeMismatchError); // TODO - we can probably always convert to f64
	}
	Ok(if a_data.shift >= b_data.shift { a } else { b })
}

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DTypeKind {
	Float,
	Int,
	Uint,
}

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DTypeId {
	U8 = 1,
	F32 = 2,
	F64 = 3,
}

impl std::fmt::Display for DType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		let data = DTypeStruct::from_dtype(*self);
		write!(f, "{:?}", data.id)
	}
}

impl DTypeId {
	pub fn to_dtype(self) -> DType {
		static map: [DType; 3] = [
			u8::dtype, //
			f32::dtype,
			f64::dtype,
		];
		map[(self as u8 as usize) - 1]
	}
}

impl From<DTypeId> for u8 {
	fn from(id: DTypeId) -> Self {
		id as u8
	}
}

//--------------------------------------------------------------------------------------------------
