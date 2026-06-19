//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ffi::c_void;
use std::hint::cold_path;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use askama::Template;
use serde::Deserialize;

use crate::device::Device;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::{Diagnostics, ErrExtra, ErrPack, KernelGeneratorError, TensorOpError};

use super::{CudaDevice, cuda_shim::CudaKernel, kernel_build};

//--------------------------------------------------------------------------------------------------

#[derive(Template)]
#[template(escape = "none", path = "attn/common.cuh")]
struct AttnCommonTemplate {
	n_heads: usize,
	heads_per_kernel: usize,
	head_dim: usize,
	q_stride: usize,
	kv_stride: usize,
	o_stride: usize,
}

#[derive(Template)]
#[template(escape = "none", path = "attn/kernel.cu")]
struct AttnKernelTemplate {}

#[derive(Template)]
#[template(escape = "none", path = "attn/meta.cu")]
struct AttnMetaTemplate {}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct AttnKernelConfig {
	pub dtype: DType,
	pub n_heads: usize,
	pub head_dim: usize,
	pub q_stride: usize,
	pub kv_stride: usize,
	pub o_stride: usize,
}

impl AttnKernelConfig {
	pub fn heads_per_kernel(self) -> usize {
		if self.head_dim == 0 {
			1
		} else {
			(128 / self.head_dim).max(1)
		}
	}

	pub fn head_group_cnt(self) -> usize {
		self.n_heads / self.heads_per_kernel()
	}

	pub fn head_group_dim(self) -> usize {
		self.heads_per_kernel() * self.head_dim
	}

	pub fn n_ops(self, seq_len: usize, window_size: usize) -> f64 {
		#[allow(clippy::cast_precision_loss)]
		let attended_tokens_per_query = if window_size > 0 {
			(window_size as f64) + 1.0
		} else {
			((seq_len as f64) + 1.0) / 2.0
		};

		#[allow(clippy::cast_precision_loss)]
		{
			4.0 * ((self.n_heads * self.head_dim) as f64) * (seq_len as f64) * attended_tokens_per_query
		}
	}

	fn validate(self, diag: &mut Diagnostics) -> Result<(), KernelGeneratorError> {
		if self.dtype != DType::Int8 {
			cold_path();
			diag.add_error(format!(
				"attention kernel config currently requires dtype i8, got {}",
				self.dtype,
			));
			return Err(KernelGeneratorError);
		}
		if self.n_heads == 0 {
			cold_path();
			diag.add_error("attention kernel config requires n_heads > 0".to_owned());
			return Err(KernelGeneratorError);
		}
		if self.head_dim == 0 || self.head_dim % 32 != 0 {
			cold_path();
			diag.add_error(format!(
				"attention kernel config requires head_dim to be a non-zero multiple of 32, got {}",
				self.head_dim,
			));
			return Err(KernelGeneratorError);
		}
		let heads_per_kernel = self.heads_per_kernel();
		if self.n_heads % heads_per_kernel != 0 {
			cold_path();
			diag.add_error(format!(
				"attention kernel config requires n_heads ({}) to be divisible by heads_per_kernel ({})",
				self.n_heads, heads_per_kernel,
			));
			return Err(KernelGeneratorError);
		}
		let q_cols = self.n_heads * self.head_dim;
		let kv_cols = 2 * q_cols;
		if self.q_stride < q_cols {
			cold_path();
			diag.add_error(format!(
				"attention kernel config requires q_stride >= n_heads * head_dim, got {} vs {}",
				self.q_stride, q_cols,
			));
			return Err(KernelGeneratorError);
		}
		if self.kv_stride < kv_cols {
			cold_path();
			diag.add_error(format!(
				"attention kernel config requires kv_stride >= 2 * n_heads * head_dim, got {} vs {}",
				self.kv_stride, kv_cols,
			));
			return Err(KernelGeneratorError);
		}
		if self.o_stride < q_cols {
			cold_path();
			diag.add_error(format!(
				"attention kernel config requires o_stride >= n_heads * head_dim, got {} vs {}",
				self.o_stride, q_cols,
			));
			return Err(KernelGeneratorError);
		}
		if (self.head_group_dim() * std::mem::size_of::<i8>()) % 128 != 0 {
			cold_path();
			diag.add_error(format!(
				"attention kernel config requires heads_per_kernel * head_dim to be 128B aligned, got {}",
				self.head_group_dim(),
			));
			return Err(KernelGeneratorError);
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct AttnKernel {
	pub name: String,
	pub dir_path: PathBuf,
	pub common_path: PathBuf,
	pub kernel_path: PathBuf,
	pub ptx_path: PathBuf,
	pub cubin_path: PathBuf,
	pub meta_path: PathBuf,
	pub meta_exe_path: PathBuf,
	pub meta_json_path: PathBuf,
	launcher: AttnKernelLauncher,
}

impl AttnKernel {
	pub fn new(
		device: Rc<CudaDevice>,
		kernel_name: impl AsRef<str>,
		config: AttnKernelConfig,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError> {
		let kernel_name = kernel_name.as_ref();
		if kernel_name.is_empty()
			|| kernel_name == "."
			|| kernel_name == ".."
			|| kernel_name.contains('/')
			|| kernel_name.contains('\\')
		{
			cold_path();
			diag.add_error(format!("invalid attention kernel name {kernel_name:?}"));
			return Err(KernelGeneratorError);
		}

		config.validate(diag)?;

		let dir_path = Path::new("cache").join("kernels").join(kernel_name);
		match std::fs::create_dir_all(&dir_path) {
			Ok(()) => {},
			Err(err) => {
				cold_path();
				diag.add_error(format!(
					"failed to create kernel cache directory {}: {err}",
					dir_path.display(),
				));
				return Err(KernelGeneratorError);
			},
		}

		let common_path = dir_path.join("common.cuh");
		let kernel_path = dir_path.join("kernel.cu");
		let ptx_path = dir_path.join("kernel.ptx");
		let cubin_path = dir_path.join("kernel.cubin");
		let meta_path = dir_path.join("meta.cu");
		let meta_exe_path = dir_path.join("meta");
		let meta_json_path = dir_path.join("meta.json");

		Self::generate_sources(config, &common_path, &kernel_path, &meta_path, diag)?;
		kernel_build::compile_kernel_ptx(&dir_path, &kernel_path, &ptx_path, diag)?;
		kernel_build::compile_kernel_cubin(&dir_path, &ptx_path, &cubin_path, diag)?;
		kernel_build::compile_meta_exe(&dir_path, &meta_path, &meta_exe_path, diag)?;
		kernel_build::run_meta_exe(&dir_path, &meta_exe_path, &meta_json_path, diag)?;

		let launcher = AttnKernelLauncher::new(device, config, &cubin_path, &meta_json_path, diag)?;

		Ok(Self {
			name: kernel_name.to_owned(),
			dir_path,
			common_path,
			kernel_path,
			ptx_path,
			cubin_path,
			meta_path,
			meta_exe_path,
			meta_json_path,
			launcher,
		})
	}

	pub fn launch(&self, args: AttnKernelArgs) -> Result<(), ErrPack<TensorOpError>> {
		self.launcher.launch(args)
	}

	pub fn n_ops(&self, seq_len: usize, window_size: usize) -> f64 {
		self.launcher.config.n_ops(seq_len, window_size)
	}

	fn generate_sources(
		config: AttnKernelConfig,
		common_path: &Path,
		kernel_path: &Path,
		meta_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		let common = AttnCommonTemplate {
			n_heads: config.n_heads,
			heads_per_kernel: config.heads_per_kernel(),
			head_dim: config.head_dim,
			q_stride: config.q_stride,
			kv_stride: config.kv_stride,
			o_stride: config.o_stride,
		};
		kernel_build::write_generated_file(
			common_path,
			common.render().unwrap_or_else(|_| todo!("render attention common template")),
			diag,
		)?;

		let kernel = AttnKernelTemplate {};
		kernel_build::write_generated_file(
			kernel_path,
			kernel.render().unwrap_or_else(|_| todo!("render attention kernel template")),
			diag,
		)?;

		let meta = AttnMetaTemplate {};
		kernel_build::write_generated_file(
			meta_path,
			meta.render().unwrap_or_else(|_| todo!("render attention metadata template")),
			diag,
		)?;

		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct AttnKernelArgs<'a> {
	pub q: &'a Tensor,
	pub kv: &'a Tensor,
	pub sink_k: &'a Tensor,
	pub sink_v: &'a Tensor,
	pub attn_temperature: &'a Tensor,
	pub maxes: Option<&'a Tensor>, // TODO - remove
	pub out: &'a Tensor,
	pub l: Option<&'a Tensor>,
	pub window_size: usize,
}

// This struct should only contain items where the source of truth naturally is the C++ CUDA code.
// We don't want to mirror these values in Rust and so will transfer them via the meta.json file.
#[derive(Deserialize)]
pub struct AttnLauncherMetadata {
	pub THREADS_PER_BLOCK: usize,
	pub SMEM_BYTES: usize,
	pub Q_PER_BLOCK: usize,
}

struct AttnKernelLauncher {
	device: Rc<CudaDevice>,
	config: AttnKernelConfig,
	metadata: AttnLauncherMetadata,
	kernel: CudaKernel,
}

impl AttnKernelLauncher {
	fn new(
		device: Rc<CudaDevice>,
		config: AttnKernelConfig,
		cubin_path: &Path,
		metadata_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError> {
		let metadata_json = match std::fs::read_to_string(metadata_path) {
			Ok(s) => s,
			Err(err) => {
				cold_path();
				let file = metadata_path.to_string_lossy();
				diag.add_error(format!("Error reading file {file}: {err}"));
				return Err(KernelGeneratorError);
			},
		};

		let metadata = match serde_json::from_str::<AttnLauncherMetadata>(&metadata_json) {
			Ok(m) => m,
			Err(err) => {
				cold_path();
				let file = metadata_path.to_string_lossy();
				diag.add_error(format!("Failed to parse JSON data from {file}: {err}"));
				return Err(KernelGeneratorError);
			},
		};

		if metadata.THREADS_PER_BLOCK == 0
			|| metadata.SMEM_BYTES == 0
			|| metadata.Q_PER_BLOCK == 0 {
			cold_path();
			let file = metadata_path.to_string_lossy();
			diag.add_error(format!("Invalid JSON metadata in {file}"));
			return Err(KernelGeneratorError);
		}

		let module = match device.stream.load_module_from_cubin(cubin_path) {
			Ok(module) => module,
			Err(err) => {
				cold_path();
				diag.add_error(format!(
					"failed to load generated attention CUDA module {}: {err}",
					cubin_path.display()
				));
				return Err(KernelGeneratorError);
			},
		};
		let kernel = match module.get_kernel("kernel", metadata.SMEM_BYTES) {
			Ok(kernel) => kernel,
			Err(err) => {
				cold_path();
				diag.add_error(format!(
					"failed to load generated attention CUDA kernel \"kernel\" from {}: {err}",
					cubin_path.display()
				));
				return Err(KernelGeneratorError);
			},
		};

		Ok(Self { device, config, metadata, kernel })
	}

	fn matrix_shape(
		tensor_name: &'static str,
		tensor: &Tensor,
		expected_shape: [Option<usize>; 2],
		expected_device: &CudaDevice,
		expected_dtype: DType,
	) -> Result<[usize; 2], ErrPack<TensorOpError>> {
		let shape = tensor.shape();
		let &[rows, cols] = shape else {
			cold_path();
			return Err(attn_launch_error(format!(
				"attention input {tensor_name} must be 2D, got shape {shape:?}"
			)));
		};
		if rows == 0 || cols == 0 {
			cold_path();
			return Err(attn_launch_error(format!(
				"attention input {tensor_name} has zero size {shape:?}"
			)));
		}
		if let Some(expected_rows) = expected_shape[0] && rows != expected_rows {
			cold_path();
			return Err(attn_launch_error(format!(
				"attention input {tensor_name} has {rows} rows, but should have {expected_rows}"
			)));
		}
		if let Some(expected_cols) = expected_shape[1] && cols != expected_cols {
			cold_path();
			return Err(attn_launch_error(format!(
				"attention input {tensor_name} has {cols} columns, but should have {expected_cols}"
			)));
		}
		Self::tensor_dtype_and_device(tensor_name, tensor, expected_device, expected_dtype)?;
		Ok([rows, cols])
	}

	fn vector_shape(
		tensor_name: &'static str,
		tensor: &Tensor,
		expected_len: usize,
		expected_device: &CudaDevice,
		expected_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>> {
		let shape = tensor.shape();
		match shape {
			&[len] if len == expected_len => {},
			&[len, 1] if len == expected_len => {},
			_ => {
				cold_path();
				return Err(attn_launch_error(format!(
					"attention input {tensor_name} must have shape [{expected_len}] or [{expected_len}, 1], got {shape:?}"
				)));
			},
		}
		Self::tensor_dtype_and_device(tensor_name, tensor, expected_device, expected_dtype)
	}

	fn tensor_dtype_and_device(
		tensor_name: &'static str,
		tensor: &Tensor,
		expected_device: &CudaDevice,
		expected_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>> {
		let dtype = tensor.dtype();
		if dtype != expected_dtype {
			cold_path();
			return Err(attn_launch_error(format!(
				"attention input {tensor_name} has dtype {dtype}, but should have {expected_dtype}"
			)));
		}
		if !tensor.is_on_device(expected_device) {
			cold_path();
			let dev: &dyn Device = expected_device;
			let dev_name = dev.name();
			return Err(attn_launch_error(format!(
				"attention input {tensor_name} is not on device {dev_name}"
			)));
		}
		Ok(())
	}

	fn launch(&self, args: AttnKernelArgs) -> Result<(), ErrPack<TensorOpError>> {
		let AttnKernelArgs {
			q,
			kv,
			sink_k,
			sink_v,
			attn_temperature,
			maxes,
			out,
			l,
			window_size,
		} = args;

		if maxes.is_some() {
			cold_path();
			return Err(attn_launch_error(
				"precomputed attention maxes are not supported by the Rust Tensor dtype model yet",
			));
		}

		let config = self.config;
		let [seq_len, _q_cols] = Self::matrix_shape(
			"q", q, [None, Some(config.q_stride)],
			&self.device, config.dtype
		)?;
		Self::matrix_shape(
			"kv", kv, [Some(seq_len), Some(config.kv_stride)],
			&self.device, config.dtype
		)?;
		Self::matrix_shape(
			"sink_k", sink_k, [Some(config.n_heads), Some(config.head_dim)],
			&self.device, config.dtype
		)?;
		Self::matrix_shape(
			"sink_v", sink_v, [Some(config.n_heads), Some(config.head_dim)],
			&self.device, config.dtype
		)?;
		Self::vector_shape(
			"attn_temperature", attn_temperature, config.n_heads,
			&self.device, DType::F32
		)?;
		Self::matrix_shape(
			"out", out, [Some(seq_len), Some(config.o_stride)],
			&self.device, config.dtype
		)?;
		if let Some(l) = l {
			Self::matrix_shape(
				"l", l, [Some(config.n_heads), Some(seq_len)],
				&self.device, DType::F32
			)?;
		}

		if seq_len % self.metadata.Q_PER_BLOCK != 0 {
			cold_path();
			return Err(attn_launch_error(format!(
				"attention seq_len must be divisible by {}, got {seq_len}",
				self.metadata.Q_PER_BLOCK,
			)));
		}
		if window_size > 0 && window_size % 16 != 0 {
			cold_path();
			return Err(attn_launch_error(format!(
				"attention window_size must be zero or divisible by 16, got {window_size}"
			)));
		}

		let mut seq_len_arg = seq_len;
		let mut q_ptr = unsafe { q.device_ptr().as_ptr::<c_void>() };
		let mut kv_ptr = unsafe { kv.device_ptr().as_ptr::<c_void>() };
		let mut sink_k_ptr = unsafe { sink_k.device_ptr().as_ptr::<c_void>() };
		let mut sink_v_ptr = unsafe { sink_v.device_ptr().as_ptr::<c_void>() };
		let mut attn_temperature_ptr = unsafe {
			attn_temperature.device_ptr().as_ptr::<c_void>()
		};
		let mut maxes_ptr = std::ptr::null_mut::<c_void>();
		let mut out_ptr = unsafe { out.device_ptr().as_ptr::<c_void>() };
		let mut l_ptr = match l {
			Some(l) => unsafe { l.device_ptr().as_ptr::<c_void>() },
			None => std::ptr::null_mut::<c_void>(),
		};
		let mut window_size_arg = window_size;

		let mut raw_args = [
			(&raw mut seq_len_arg).cast::<c_void>(),
			(&raw mut q_ptr).cast::<c_void>(),
			(&raw mut kv_ptr).cast::<c_void>(),
			(&raw mut sink_k_ptr).cast::<c_void>(),
			(&raw mut sink_v_ptr).cast::<c_void>(),
			(&raw mut attn_temperature_ptr).cast::<c_void>(),
			(&raw mut maxes_ptr).cast::<c_void>(),
			(&raw mut out_ptr).cast::<c_void>(),
			(&raw mut l_ptr).cast::<c_void>(),
			(&raw mut window_size_arg).cast::<c_void>(),
		];
		unsafe {
			self.kernel.launch(
				&self.device.stream,
				[seq_len / self.metadata.Q_PER_BLOCK, config.head_group_cnt(), 1],
				[self.metadata.THREADS_PER_BLOCK, 1, 1],
				self.metadata.SMEM_BYTES,
				&mut raw_args,
			)
		}
	}
}

fn attn_launch_error(message: impl Into<String>) -> ErrPack<TensorOpError> {
	ErrPack {
		code: TensorOpError::Other,
		extra: Some(Box::new(ErrExtra {
			message: message.into().into(),
			nested: None,
		})),
	}
}

//--------------------------------------------------------------------------------------------------
