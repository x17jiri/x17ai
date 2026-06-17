use askama::Template;
use std::ffi::c_void;
use std::hint::cold_path;
use std::num::NonZeroUsize;
use std::path::Path;
use serde::Deserialize;

use crate::device::Device;
use crate::device::cuda::cuda_shim::CudaKernel;
use crate::device::cuda::{CudaDevice, GemmKernelArgs, GemmKernelExtraArgs, GemmKernelLauncher};
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::{ErrExtra, ErrPack, TensorOpError};

pub struct BasicGemmWriterTemplate {
	pub use_l2_norm: bool,
	pub use_geglu: bool,
	pub c_type: &'static str,
	pub c_stride_expr: &'static str,
	pub output_cols_divisor: usize,
	pub scale_val: String,
	pub scale_dscr: String,
	pub head_dim: usize,
	pub sep_dim: usize,
	pub eps_val: String,
	pub head_scale_val: String,
	pub head_scale_dscr: String,
	pub sep_scale_val: String,
	pub sep_scale_dscr: String,
	pub geglu_inp_scale_val: String,
	pub geglu_inp_scale_dscr: String,
	pub geglu_out_scale_val: String,
	pub geglu_out_scale_dscr: String,
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
pub struct BasicGemmMetaTemplate {}

#[derive(Deserialize)]
pub struct BasicGemmMetadata {
	pub THREADS_PER_BLOCK: usize,
	pub SMEM_BYTES: usize,
	pub M_PER_BLOCK: usize,
	pub N_PER_BLOCK: usize,
}

pub struct BasicGemmLaunchInfo {
	pub a_cols: usize,
	pub b_rows: Option<NonZeroUsize>,
	pub output_cols_divisor: usize,
	pub head_dim: usize,
	pub sep_dim: usize,
	pub has_rrms_output: bool,
	pub a_dtype: DType,
	pub b_dtype: DType,
	pub c_dtype: DType,
}

pub struct BasicGemmKernelLauncher {
	pub metadata: BasicGemmMetadata,
	pub kernel: CudaKernel,
	pub launch_info: BasicGemmLaunchInfo,
}

impl BasicGemmKernelLauncher {
	pub fn new(
		device: &CudaDevice,
		cubin_path: &Path,
		config_path: &Path,
		launch_info: BasicGemmLaunchInfo,
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

		if metadata.THREADS_PER_BLOCK == 0
			|| metadata.M_PER_BLOCK == 0
			|| metadata.N_PER_BLOCK == 0
			|| launch_info.a_cols == 0
			|| launch_info.output_cols_divisor == 0
			|| metadata.N_PER_BLOCK % launch_info.output_cols_divisor != 0
			|| (launch_info.has_rrms_output && launch_info.head_dim == 0)
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

		Ok(Self { metadata, kernel, launch_info })
	}

	fn matrix_shape<'a>(
		tensor_name: &'static str,
		tensor: &'a Tensor,
		expected_shape: [Option<usize>; 2],
		expected_device: &CudaDevice,
		expected_dtype: DType,
	) -> Result<[usize; 2], ErrPack<TensorOpError>> {
		let shape = tensor.shape();
		let &[rows, cols] = shape else {
			cold_path();
			return Err(gemm_launch_error(format!(
				"GEMM input {tensor_name} must be 2D, got shape {shape:?}"
			)));
		};
		if rows == 0 || cols == 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"GEMM input {tensor_name} has zero size {shape:?}"
			)));
		}
		let dtype = tensor.dtype();
		if  dtype != expected_dtype {
			cold_path();
			return Err(gemm_launch_error(format!(
				"Invalid dtype. Expected {expected_dtype}, got {dtype}"
			)));
		}
		if !tensor.is_on_device(expected_device) {
			cold_path();
			let dev: &dyn Device = expected_device;
			let dev_name = dev.name();
			return Err(gemm_launch_error(format!(
				"GEMM input {tensor_name} is not on device {dev_name}"
			)));
		}
		Ok([rows, cols])
	}
}

impl GemmKernelLauncher for BasicGemmKernelLauncher {
	fn launch(
		&self, device: &CudaDevice, args: GemmKernelArgs<'_>
	) -> Result<(), ErrPack<TensorOpError>> {
		let a = args.a;
		let b = args.b;
		let c = args.c;

		let [a_rows, a_cols] = Self::matrix_shape(
			"a", a, [None, Some(self.launch_info.a_cols)],
			device, self.launch_info.a_dtype
		)?;
		let [b_rows, b_cols] = Self::matrix_shape(
			"c", b, [self.launch_info.b_rows, Some(self.launch_info.a_cols)],
			device, self.launch_info.b_dtype
		)?;
		let [c_rows, c_cols] = Self::matrix_shape(
			"c", c, [Some(a_rows / divisor), Some(b_rows)],
			device, self.launch_info.c_dtype
		)?;

		if a_cols != self.launch_info.a_cols {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input a has {} columns, but kernel was compiled for {}",
				a_cols,
				self.launch_info.a_cols
			)));
		}
		if b_cols != self.launch_info.a_cols {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input b has {} columns, but kernel was compiled for {}",
				b_cols,
				self.launch_info.a_cols
			)));
		}
		if let Some(B_ROWS) = self.launch_info.b_rows && b_rows != B_ROWS.get() {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM input b has {b_rows} rows, but kernel was compiled for {B_ROWS}",
			)));
		}

		// TODO - assume M_PER_BLOCK and N_PER_BLOCK are powers of 2, which mean division is shift
		let c_cols_pre_epilogue = b_rows;
		if c_rows % self.metadata.M_PER_BLOCK != 0
			|| c_cols_pre_epilogue % self.metadata.N_PER_BLOCK != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"basic GEMM output shape [{c_rows}, {c_cols_pre_epilogue}] must be divisible by tile shape [{}, {}]",
				self.metadata.M_PER_BLOCK, self.metadata.N_PER_BLOCK
			)));
		}

		if c_rows != a_rows
			|| c_cols * self.launch_info.output_cols_divisor != c_cols_pre_epilogue {
			cold_path();
			let expected_c_cols = c_cols_pre_epilogue / self.launch_info.output_cols_divisor;
			return Err(gemm_launch_error(format!(
				"basic GEMM output c has shape [{c_rows}, {c_cols}], expected [{a_rows}, {expected_c_cols}]",
			)));
		}

		let grid_x = a_rows / self.metadata.M_PER_BLOCK;
		let grid_y = b_rows / self.metadata.N_PER_BLOCK;
		let mut a_ptr = unsafe { a.device_ptr().as_ptr::<c_void>() };
		let mut a_rows_arg = a_rows;
		let mut b_ptr = unsafe { b.device_ptr().as_ptr::<c_void>() };
		let mut b_rows_arg = b_rows;
		let mut c_ptr = unsafe { c.device_ptr().as_ptr::<c_void>() };

		match (self.launch_info.has_rrms_output, args.extra) {
			(false, GemmKernelExtraArgs::None) => {
				let mut raw_args = [
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
						&mut raw_args,
					)
				}
			},
			(true, GemmKernelExtraArgs::RMSNorm { rrms }) => {
				self.validate_rrms(device, rrms, a_rows, b_rows)?;
				let mut rrms_ptr = unsafe { rrms.device_ptr().as_ptr::<c_void>() };
				let mut raw_args = [
					(&raw mut a_ptr).cast::<c_void>(),
					(&raw mut a_rows_arg).cast::<c_void>(),
					(&raw mut b_ptr).cast::<c_void>(),
					(&raw mut b_rows_arg).cast::<c_void>(),
					(&raw mut c_ptr).cast::<c_void>(),
					(&raw mut rrms_ptr).cast::<c_void>(),
				];
				unsafe {
					self.kernel.launch(
						&device.stream,
						[grid_x, grid_y, 1],
						[self.metadata.THREADS_PER_BLOCK, 1, 1],
						self.metadata.SMEM_BYTES,
						&mut raw_args,
					)
				}
			},
			(true, GemmKernelExtraArgs::None) => {
				cold_path();
				Err(gemm_launch_error("basic GEMM kernel requires RMSNorm rrms output"))
			},
			(false, GemmKernelExtraArgs::RMSNorm { .. }) => {
				cold_path();
				Err(gemm_launch_error("basic GEMM kernel does not take RMSNorm rrms output"))
			},
		}
	}

	fn n_ops(&self, a: &Tensor, b: &Tensor) -> f64 {
		#![allow(clippy::cast_precision_loss)]
		2.0 * (a.size(-2) as f64) * (a.size(-1) as f64) * (b.size(-2) as f64)
	}
}

impl BasicGemmKernelLauncher {
	fn validate_rrms(
		&self,
		device: &CudaDevice,
		rrms: &Tensor,
		a_rows: usize,
		b_rows: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		if !rrms.is_on_device(device) {
			cold_path();
			let dev: &dyn Device = device;
			let dev_name = dev.name();
			return Err(gemm_launch_error(format!("RMSNorm rrms output needs to be on device {dev_name}")));
		}
		if rrms.dtype() != DType::F32 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"RMSNorm rrms output must be f32, got {}",
				rrms.dtype()
			)));
		}

		let chunk = self.launch_info.head_dim + self.launch_info.sep_dim;
		if chunk == 0 || b_rows % chunk != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"RMSNorm metadata dimensions [{}, {}] do not divide b rows {b_rows}",
				self.launch_info.head_dim,
				self.launch_info.sep_dim,
			)));
		}
		let rrms_cols = b_rows / chunk;
		let rrms_shape = rrms.shape();
		if rrms_shape != [a_rows, rrms_cols] {
			cold_path();
			return Err(gemm_launch_error(format!(
				"RMSNorm rrms output has shape {rrms_shape:?}, expected [{a_rows}, {rrms_cols}]"
			)));
		}
		Ok(())
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
