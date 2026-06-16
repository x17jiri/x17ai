use askama::Template;
use std::ffi::c_void;
use std::hint::cold_path;
use std::num::NonZeroUsize;
use std::path::Path;
use serde::Deserialize;

use crate::device::Device;
use crate::device::cuda::cuda_shim::CudaKernel;
use crate::device::cuda::{CudaDevice, GemmKernelLauncher};
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::{ErrExtra, ErrPack, TensorOpError};

pub struct BasicGemmWriterTemplate {
	pub use_l2_norm: bool,
	pub scale_val: String,
	pub scale_dscr: String,
	pub head_dim: usize,
	pub sep_dim: usize,
	pub eps_val: String,
	pub head_scale_val: String,
	pub head_scale_dscr: String,
	pub sep_scale_val: String,
	pub sep_scale_dscr: String,
	pub has_rrms_output: bool,
}

#[derive(Template)]
#[template(escape = "none", path = "basic_gemm/common.cuh")]
pub struct BasicGemmCommonTemplate<'a> {
	pub a_cols: usize,
	pub b_rows: Option<NonZeroUsize>,
	pub writer: &'a BasicGemmWriterTemplate,
}

#[derive(Template)]
#[template(escape = "none", path = "basic_gemm/kernel.cu")]
pub struct BasicGemmKernelTemplate<'a> {
	pub a_cols: usize,
	pub b_rows: Option<NonZeroUsize>,
	pub writer: &'a BasicGemmWriterTemplate,
}

#[derive(Template)]
#[template(escape = "none", path = "basic_gemm/meta.cu")]
pub struct BasicGemmMetaTemplate {
	pub a_cols: usize,
	pub b_rows: Option<NonZeroUsize>,
}

#[derive(Deserialize)]
pub struct BasicGemmMetadata {
	pub A_COLS: usize,
	pub B_ROWS: Option<NonZeroUsize>,
	pub THREADS_PER_BLOCK: usize,
	pub SMEM_BYTES: usize,
	pub M_PER_BLOCK: usize,
	pub N_PER_BLOCK: usize,
}

pub struct BasicGemmKernelLauncher {
	pub metadata: BasicGemmMetadata,
	pub kernel: CudaKernel,
}

impl BasicGemmKernelLauncher {
	pub fn new(
		device: &CudaDevice,
		cubin_path: &Path,
		config_path: &Path
	) -> Result<Self, ErrPack<TensorOpError>> {
		let config = match std::fs::read_to_string(config_path) {
			Ok(s) => s,
			Err(err) => {
				cold_path();
				let file = config_path.to_string_lossy();
				return Err(ErrPack {
					code: TensorOpError::KernelGenerator,
					extra: Some(Box::new(ErrExtra {
						message: format!("Error reading file {file}").into(),
						nested: Some(Box::new(err))
					})),
				});
			},
		};

		let metadata = match serde_json::from_str::<BasicGemmMetadata>(&config) {
			Ok(m) => m,
			Err(err) => {
				cold_path();
				let file = config_path.to_string_lossy();
				return Err(ErrPack {
					code: TensorOpError::KernelGenerator,
					extra: Some(Box::new(ErrExtra {
						message: format!("Failed to parse JSON data from {file}").into(),
						nested: Some(Box::new(err))
					})),
				});
			},
		};

		if metadata.A_COLS == 0
			|| metadata.THREADS_PER_BLOCK == 0
			|| metadata.M_PER_BLOCK == 0
			|| metadata.N_PER_BLOCK == 0
		{
			cold_path();
			let file = config_path.to_string_lossy();
			return Err(ErrPack {
				code: TensorOpError::KernelGenerator,
				extra: Some(Box::new(ErrExtra {
					message: format!("Invalid JSON metadata in {file}").into(),
					nested: None
				})),
			});
		}

		let kernel = device.stream
			.load_module_from_cubin(cubin_path)?
			.get_kernel("kernel", metadata.SMEM_BYTES)?;

		Ok(Self { metadata, kernel })
	}
}

impl GemmKernelLauncher for BasicGemmKernelLauncher {
	fn launch(
		&self, device: &CudaDevice, a: &Tensor, b: &Tensor, c: &Tensor
	) -> Result<(), ErrPack<TensorOpError>> {
		if !a.is_on_device(device) || !b.is_on_device(device) || !c.is_on_device(device) {
			cold_path();
			let dev: &dyn Device = device;
			let dev_name = dev.name();
			return Err(gemm_launch_error(format!("all input tensors need to be on device {dev_name}")));
		}

		// TODO - we want to support other types. Need to be able to check them
		if a.dtype() != DType::Int8 || b.dtype() != DType::Int8 || c.dtype() != DType::Int8 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM inputs a must be i8, got {}",
				a.dtype()
			)));
		}

		let a_shape = a.shape();
		let &[a_rows, a_cols] = a_shape else {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input a must be 2D, got shape {a_shape:?}"
			)));
		};
		let b_shape = b.shape();
		let &[b_rows, b_cols] = b_shape else {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input b must be 2D, got shape {b_shape:?}"
			)));
		};
		let c_shape = c.shape();
		let &[c_rows, c_cols] = c_shape else {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM output c must be 2D, got shape {c_shape:?}"
			)));
		};

		if a_rows == 0 || b_rows == 0 {
			cold_path();
			return Err(gemm_launch_error("basic GEMM does not support zero size tensors"));
		}
		if a_cols != self.metadata.A_COLS {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input a has {} columns, but kernel was compiled for {}",
				a_cols,
				self.metadata.A_COLS
			)));
		}
		if b_cols != self.metadata.A_COLS {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input b has {} columns, but kernel was compiled for {}",
				b_cols,
				self.metadata.A_COLS
			)));
		}
		if let Some(B_ROWS) = self.metadata.B_ROWS && b_rows != B_ROWS.get() {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input b has {b_rows} rows, but kernel was compiled for {B_ROWS}",
			)));
		}
		if c_rows != a_rows || c_cols != b_rows {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM output c has shape {c_shape:?}, expected [{a_rows}, {b_rows}]",
			)));
		}

		if c_rows % self.metadata.M_PER_BLOCK != 0 || c_cols % self.metadata.N_PER_BLOCK != 0{
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM output shape [{c_rows}, {c_cols}] must be divisible by tile shape [{}, {}]",
				self.metadata.M_PER_BLOCK, self.metadata.N_PER_BLOCK
			)));
		}

		let grid_x = a_rows / self.metadata.M_PER_BLOCK;
		let grid_y = b_rows / self.metadata.N_PER_BLOCK;
		let mut a_ptr = unsafe { a.device_ptr().as_ptr::<c_void>() };
		let mut a_rows_arg = a_rows;
		let mut b_ptr = unsafe { b.device_ptr().as_ptr::<c_void>() };
		let mut b_rows_arg = b_rows;
		let mut c_ptr = unsafe { c.device_ptr().as_ptr::<c_void>() };
		let mut args = [
			(&raw mut a_ptr).cast::<c_void>(),
			(&raw mut a_rows_arg).cast::<c_void>(),
			(&raw mut b_ptr).cast::<c_void>(),
			(&raw mut b_rows_arg).cast::<c_void>(),
			(&raw mut c_ptr).cast::<c_void>(),
		];

		unsafe {
			self.kernel.launch(
				&device.stream,
				[grid_x, grid_y, 1],
				[self.metadata.THREADS_PER_BLOCK, 1, 1],
				self.metadata.SMEM_BYTES,
				&mut args,
			)
		}
	}

	fn n_ops(&self, a: &Tensor, b: &Tensor) -> f64 {
		#![allow(clippy::cast_precision_loss)]
		2.0 * (a.size(-2) as f64) * (a.size(-1) as f64) * (b.size(-2) as f64)
	}
}

fn gemm_launch_error(message: impl Into<String>) -> ErrPack<TensorOpError> {
	ErrPack {
		code: TensorOpError::Other,
		extra: Some(Box::new(ErrExtra {
			message: message.into().into(),
			nested: None,
		})),
	}
}
