//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::num::NonZeroU32;

//--------------------------------------------------------------------------------------------------

pub const MAX_DTYPE_ALIGN: usize = 8; // 64-bit

pub trait HasDType {
	const dtype: DType;
}

impl HasDType for u8 {
	const dtype: DType = DType::from_struct(DTypeStruct {
		kind: DTypeKind::Uint,
		bits: 8,
		bytes: 1,
		id: DTypeId::U8,
	});
}

impl HasDType for f32 {
	const dtype: DType = DType::from_struct(DTypeStruct {
		kind: DTypeKind::Float,
		bits: 32,
		bytes: 4,
		id: DTypeId::F32,
	});
}

impl HasDType for f64 {
	const dtype: DType = DType::from_struct(DTypeStruct {
		kind: DTypeKind::Float,
		bits: 64,
		bytes: 8,
		id: DTypeId::F64,
	});
}

pub struct DTypeStruct {
	kind: DTypeKind,
	bits: u8,
	bytes: u8,
	id: DTypeId,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[repr(transparent)]
pub struct DType(NonZeroU32);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DTypeMismatch;

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
			bits: bytes[1],
			bytes: bytes[2],
			id: unsafe { std::mem::transmute(bytes[3]) },
		}
	}
}

impl DType {
	pub const fn from_struct(data: DTypeStruct) -> DType {
		let val = u32::from_le_bytes([data.kind as u8, data.bits, data.bytes, data.id as u8]);
		// SAFETY: DTypeId starts at 1, so val is never 0
		DType(unsafe { NonZeroU32::new_unchecked(val) })
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
		usize::from(data.bits)
	}

	/// NOTE: We don't support types with size 0.
	/// However, this function will return 0 if the type uses 1, 2 or 4 bits.
	pub fn bytes(self) -> usize {
		let data = DTypeStruct::from_dtype(self);
		usize::from(data.bytes)
	}

	pub fn align(self) -> usize {
		let data = DTypeStruct::from_dtype(self);
		usize::from(data.bytes).max(1)
	}

	pub fn array_bytes(self, elems: usize) -> Option<usize> {
		let data = DTypeStruct::from_dtype(self);
		debug_assert!(data.bits.is_power_of_two());
		if data.bytes < 1 {
			todo!("bitfields");
		}
		usize::from(data.bytes).checked_mul(elems)
	}

	/// # Safety
	///
	/// `elems` must be such that the total size in bytes does not overflow.
	pub unsafe fn array_bytes_unchecked(self, elems: usize) -> usize {
		let data = DTypeStruct::from_dtype(self);
		debug_assert!(data.bits.is_power_of_two());
		if data.bytes < 1 {
			todo!("bitfields");
		}
		usize::from(data.bytes) * elems
	}
}

pub fn common_dtype(a: DType, b: DType) -> Result<DType, DTypeMismatch> {
	let a_data = DTypeStruct::from_dtype(a);
	let b_data = DTypeStruct::from_dtype(b);
	if a_data.kind != b_data.kind {
		cold_path();
		return Err(DTypeMismatch); // TODO - we can probably always convert to f64
	}
	Ok(if a_data.bits >= b_data.bits { a } else { b })
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
