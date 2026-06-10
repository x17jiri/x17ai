//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;
use std::rc::Rc;

use askama::Template;

use crate::dtype::DType;
use crate::{DeviceAllocError, ErrPack, TensorOpError};

use super::cuda_shim::CudaStream;
use super::{Device, DevicePtr};

//--------------------------------------------------------------------------------------------------

pub struct CudaDevice {
	name: String,
	stream: CudaStream,
}

impl CudaDevice {
	pub fn new(device_id: usize) -> Result<Rc<Self>, ErrPack<TensorOpError>> {
		Self::new_named(device_id, format!("CUDA:{device_id}"))
	}

	pub fn new_named(
		device_id: usize,
		name: String,
	) -> Result<Rc<Self>, ErrPack<TensorOpError>> {
		let stream = CudaStream::new(device_id)?;
		Ok(Rc::new(Self { name, stream }))
	}
}

impl Device for CudaDevice {
	fn name(&self) -> &str {
		&self.name
	}

	unsafe fn new_buffer(&self, bytes: usize) -> Result<DevicePtr, DeviceAllocError> {
		unsafe { self.stream.alloc(bytes).map_err(|_| DeviceAllocError) }
	}

	unsafe fn drop_buffer(&self, device_ptr: DevicePtr, _bytes: usize) {
		unsafe { self.stream.free(device_ptr) };
	}

	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: DevicePtr,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { self.stream.upload_data(src, dst, 0, bytes) }
	}

	unsafe fn download_data(
		&self,
		src: DevicePtr,
		dst: NonNull<u8>,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { self.stream.download_data(src, dst, 0, bytes) }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RMSNormEpulogue {
	pub head_dim: usize,
	pub sep_dim: usize,

	pub head_scale: f64,
	pub sep_scale: f64,
}

pub struct ResidualEpilogue {
	pub old_scale: f64,
	pub new_scale: f64,
	pub out_scale: f64,
}

pub struct GeGluEpilogue {
	pub inp_scale: f64,
	pub out_scale: f64,
}

pub enum GemmEpilogue {
	Scale(f64),
	RMSNorm(RMSNormEpulogue),
	Residual(ResidualEpilogue),
	GeGlu(GeGluEpilogue),
}

pub struct GemmInput {
	pub dtype: DType,
	pub cols: usize,
	pub trans: bool,
}

pub struct GemmKernelConfig {
	pub a: GemmInput,
	pub b: GemmInput,

	pub epilogue: GemmEpilogue,
	pub c_dtype: DType,
}

#[derive(Template)]
#[template(escape = "none", path = "gemm_kernel.cu")]
struct GemmKernelTemplate {
	namespace_name: &'static str,
	launcher_name: &'static str,
	a_cols: usize,
	c_cols: usize,
	scale: String,
}

pub fn generate_gemm_kernel(config: &GemmKernelConfig) -> String {
	if config.a.dtype != DType::Int8 {
		todo!("support GEMM inputs other than i8");
	}
	if config.b.dtype != DType::Int8 {
		todo!("support GEMM weights other than i8");
	}
	if config.c_dtype != DType::Int8 {
		todo!("support GEMM outputs other than i8");
	}
	if config.a.trans {
		todo!("support transposed GEMM input A");
	}
	if !config.b.trans {
		todo!("support non-transposed GEMM input B");
	}
	if config.a.cols == 0 || config.b.cols == 0 {
		todo!("support empty GEMM dimensions");
	}

	let scale = match config.epilogue {
		GemmEpilogue::Scale(scale) => scale,
		_ => todo!("support GEMM epilogues other than scale"),
	};
	if !scale.is_finite() {
		todo!("support non-finite GEMM scale values");
	}

	let template = GemmKernelTemplate {
		namespace_name: "X17GeneratedGemm",
		launcher_name: "x17ai_gemm_i8_launch",
		a_cols: config.a.cols,
		c_cols: config.b.cols,
		scale: format!("{scale:.17e}"),
	};

	template.render().unwrap_or_else(|_| todo!("render GEMM kernel template"))
}

//--------------------------------------------------------------------------------------------------
