
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DType {
	E4m3 = 0x30,
	Int8 = 0x31,
	Bf16 = 0x40,
	F32 = 0x50,
}

impl DType {
	pub fn bits(self) -> usize {
		return 1_usize << ((self as u8) >> 4);
	}
}

pub trait HasDType {
	const dtype: DType;
}

#[derive(Clone, Copy, Debug)]
pub struct UnsupportedDTypeError;

impl HasDType for f32 {
	const dtype: DType = DType::F32;
}

impl TryFrom<safetensors::tensor::Dtype> for DType {
	type Error = UnsupportedDTypeError;

	fn try_from(dtype: safetensors::tensor::Dtype) -> Result<Self, Self::Error> {
		match dtype {
			safetensors::tensor::Dtype::F8_E4M3 => Ok(Self::E4m3),
			safetensors::tensor::Dtype::I8 => Ok(Self::Int8),
			safetensors::tensor::Dtype::BF16 => Ok(Self::Bf16),
			safetensors::tensor::Dtype::F32 => Ok(Self::F32),
			_ => Err(UnsupportedDTypeError),
		}
	}
}

impl std::fmt::Display for DType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			Self::E4m3 => write!(f, "e4m3"),
			Self::Int8 => write!(f, "i8"),
			Self::Bf16 => write!(f, "bf16"),
			Self::F32 => write!(f, "f32"),
		}
	}
}
