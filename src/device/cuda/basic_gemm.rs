use askama::Template;
use std::{ffi::c_void, hint::cold_path, path::Path};
use serde::{Deserialize};
use crate::{
	device::cuda::{cuda_shim::CudaKernel, CudaDevice, GemmKernelLauncher},
	dtype::DType,
	tensor::Tensor,
	ErrExtra, ErrPack, TensorOpError,
};

#[derive(Template)]
#[template(escape = "none", path = "basic_gemm/common.cuh")]
pub struct BasicGemmCommonTemplate<'a> {
	pub a_cols: usize,
	pub b_rows: Option<usize>,
	pub scale_val: String,
	pub scale_dscr: &'a str,
}

#[derive(Template)]
#[template(escape = "none", path = "basic_gemm/kernel.cu")]
pub struct BasicGemmKernelTemplate {
	pub b_rows: Option<usize>,
}

#[derive(Template)]
#[template(escape = "none", path = "basic_gemm/meta.cu")]
pub struct BasicGemmMetaTemplate;

#[derive(Deserialize)]
pub struct BasicGemmMetadata {
	pub A_COLS: usize,
	pub B_ROWS: usize,
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
		if a.is_on_cpu() {
			cold_path();
			return Err(gemm_launch_error("basic GEMM input a is on CPU"));
		}
		if b.is_on_cpu() {
			cold_path();
			return Err(gemm_launch_error("basic GEMM input b is on CPU"));
		}
		if c.is_on_cpu() {
			cold_path();
			return Err(gemm_launch_error("basic GEMM output c is on CPU"));
		}
		if a.dtype() != DType::Int8 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input a must be i8, got {}",
				a.dtype()
			)));
		}
		if b.dtype() != DType::Int8 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input b must be i8, got {}",
				b.dtype()
			)));
		}
		if c.dtype() != DType::Int8 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM output c must be i8, got {}",
				c.dtype()
			)));
		}

		let a_shape = a.shape();
		let b_shape = b.shape();
		let c_shape = c.shape();
		if a_shape.len() != 2 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input a must be 2D, got shape {:?}",
				a_shape
			)));
		}
		if b_shape.len() != 2 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input b must be 2D, got shape {:?}",
				b_shape
			)));
		}
		if c_shape.len() != 2 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM output c must be 2D, got shape {:?}",
				c_shape
			)));
		}

		let a_rows = a_shape[0];
		let a_cols = a_shape[1];
		let b_rows = b_shape[0];
		let b_cols = b_shape[1];
		if a_rows == 0 || a_cols == 0 || b_rows == 0 {
			cold_path();
			return Err(gemm_launch_error("basic GEMM does not support empty tensors"));
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
		if b_rows != self.metadata.B_ROWS {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input b has {} rows, but kernel was compiled for {}",
				b_rows,
				self.metadata.B_ROWS
			)));
		}
		if c_shape[0] != a_rows || c_shape[1] != b_rows {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM output c has shape {:?}, expected [{}, {}]",
				c_shape,
				a_rows,
				b_rows
			)));
		}

		if self.metadata.THREADS_PER_BLOCK == 0
			|| self.metadata.M_PER_BLOCK == 0
			|| self.metadata.N_PER_BLOCK == 0
		{
			cold_path();
			return Err(gemm_launch_error("basic GEMM metadata contains a zero launch dimension"));
		}
		if a_rows % self.metadata.M_PER_BLOCK != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input a rows ({}) must be divisible by M_PER_BLOCK ({})",
				a_rows,
				self.metadata.M_PER_BLOCK
			)));
		}
		if b_rows % self.metadata.N_PER_BLOCK != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input b rows ({}) must be divisible by N_PER_BLOCK ({})",
				b_rows,
				self.metadata.N_PER_BLOCK
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
			(&mut a_ptr as *mut *mut c_void).cast::<c_void>(),
			(&mut a_rows_arg as *mut usize).cast::<c_void>(),
			(&mut b_ptr as *mut *mut c_void).cast::<c_void>(),
			(&mut b_rows_arg as *mut usize).cast::<c_void>(),
			(&mut c_ptr as *mut *mut c_void).cast::<c_void>(),
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
