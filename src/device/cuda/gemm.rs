//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::ffi::c_void;
use std::hint::cold_path;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::rc::Rc;

use askama::Template;
use serde::Deserialize;

use crate::device::cuda::gemm_templates::{BasicGemmCommonTemplate, BasicGemmKernelTemplate, BasicGemmMetaTemplate, BasicGemmWriterTemplate};
use crate::{Diagnostics, ErrExtra, ErrPack, KernelGeneratorError, TensorOpError};
use crate::device::Device;
use crate::dtype::DType;
use crate::tensor::Tensor;

use super::{CudaDevice, cuda_shim::CudaKernel};

//--------------------------------------------------------------------------------------------------

pub struct Scale {
	pub value: f64,
	pub description: Cow<'static, str>,
}

pub struct ScaleConfig(pub Scale);

pub struct RMSNormConfig {
	pub eps: f64,
	pub head_dim: usize,
	pub sep_dim: usize,

	pub head_scale: Scale,
	pub sep_scale: Scale,
}

pub struct ResidualConfig {
	pub old_scale: Scale,
	pub new_scale: Scale,
	pub out_scale: Scale,
}

pub struct GeGluConfig {
	pub inp_scale: Scale,
	pub out_scale: Scale,
}

//--------------------------------------------------------------------------------------------------

pub struct GemmInputConfig {
	pub dtype: DType,
	pub cols: usize,
	pub rows: Option<NonZeroUsize>,
	pub trans: bool,
}

pub struct GemmConfig {
	pub a: GemmInputConfig,
	pub b: GemmInputConfig,
	pub c_dtype: DType,
}

pub trait EpilogueConfig {
	fn is_c_dtype_allowed(&self, c_dtype: DType) -> bool;
	fn writer_template(&self, b_rows: Option<NonZeroUsize>) -> BasicGemmWriterTemplate;
}

impl EpilogueConfig for ScaleConfig {
	fn is_c_dtype_allowed(&self, c_dtype: DType) -> bool {
		c_dtype == DType::Int8
	}

	fn writer_template(&self, _b_rows: Option<NonZeroUsize>) -> BasicGemmWriterTemplate {
		if !self.0.value.is_finite() {
			todo!("support non-finite GEMM scale values");
		}

		BasicGemmWriterTemplate {
			use_l2_norm: false,
			use_geglu: false,
			c_type: "b8::FixedI8",
			c_stride_expr: "b_rows",
			scale_val: format_cpp_f64(self.0.value),
			scale_dscr: self.0.description.as_ref().to_owned(),
			head_dim: 0,
			sep_dim: 0,
			eps_val: String::new(),
			head_scale_val: String::new(),
			head_scale_dscr: String::new(),
			sep_scale_val: String::new(),
			sep_scale_dscr: String::new(),
			geglu_inp_scale_val: String::new(),
			geglu_inp_scale_dscr: String::new(),
			geglu_out_scale_val: String::new(),
			geglu_out_scale_dscr: String::new(),
			has_rrms_output: false,
		}
	}
}

impl EpilogueConfig for RMSNormConfig {
	fn is_c_dtype_allowed(&self, c_dtype: DType) -> bool {
		c_dtype == DType::Int8
	}

	fn writer_template(&self, b_rows: Option<NonZeroUsize>) -> BasicGemmWriterTemplate {
		if !self.eps.is_finite() || self.eps < 0.0 {
			todo!("support invalid RMSNorm eps values");
		}
		if self.head_dim == 0 || self.sep_dim == 0 {
			todo!("support empty RMSNorm GEMM epilogue dimensions");
		}
		if self.head_dim != self.sep_dim {
			todo!("support RMSNorm GEMM epilogues where head_dim != sep_dim");
		}
		if self.head_dim % 32 != 0 || self.sep_dim % 32 != 0 {
			todo!("support RMSNorm GEMM epilogue dimensions that are not multiples of 32");
		}
		if !self.head_scale.value.is_finite() || !self.sep_scale.value.is_finite() {
			todo!("support non-finite RMSNorm GEMM scale values");
		}
		let Some(b_rows) = b_rows else {
			todo!("support RMSNorm GEMM outputs with runtime column count");
		};
		let chunk = self.head_dim + self.sep_dim;
		if b_rows.get() % chunk != 0 {
			todo!("support RMSNorm GEMM output columns that are not divisible by head_dim + sep_dim");
		}

		#[allow(clippy::cast_precision_loss)]
		let head_scale = self.head_scale.value * f64::sqrt(self.head_dim as f64);
		BasicGemmWriterTemplate {
			use_l2_norm: true,
			use_geglu: false,
			c_type: "b8::FixedI8",
			c_stride_expr: "b_rows",
			scale_val: String::new(),
			scale_dscr: String::new(),
			head_dim: self.head_dim,
			sep_dim: self.sep_dim,
			eps_val: format_cpp_f64(self.eps),
			head_scale_val: format_cpp_f64(head_scale),
			head_scale_dscr: format!(
				"({}) * sqrt({})",
				self.head_scale.description.as_ref(),
				self.head_dim,
			),
			sep_scale_val: format_cpp_f64(self.sep_scale.value),
			sep_scale_dscr: self.sep_scale.description.as_ref().to_owned(),
			geglu_inp_scale_val: String::new(),
			geglu_inp_scale_dscr: String::new(),
			geglu_out_scale_val: String::new(),
			geglu_out_scale_dscr: String::new(),
			has_rrms_output: true,
		}
	}
}

impl EpilogueConfig for ResidualConfig {
	fn is_c_dtype_allowed(&self, c_dtype: DType) -> bool {
		c_dtype == DType::Int8
	}

	fn writer_template(&self, _b_rows: Option<NonZeroUsize>) -> BasicGemmWriterTemplate {
		todo!("support residual GEMM epilogues")
	}
}

impl EpilogueConfig for GeGluConfig {
	fn is_c_dtype_allowed(&self, c_dtype: DType) -> bool {
		c_dtype == DType::E4m3
	}

	fn writer_template(&self, b_rows: Option<NonZeroUsize>) -> BasicGemmWriterTemplate {
		if !self.inp_scale.value.is_finite() || self.inp_scale.value <= 0.0 {
			todo!("support invalid GeGLU input scale values");
		}
		if !self.out_scale.value.is_finite() || self.out_scale.value <= 0.0 {
			todo!("support invalid GeGLU output scale values");
		}
		let Some(b_rows) = b_rows else {
			todo!("support GeGLU GEMM outputs with runtime column count");
		};
		if b_rows.get() % 2 != 0 {
			todo!("support GeGLU GEMM outputs with odd pregate column count");
		}

		BasicGemmWriterTemplate {
			use_l2_norm: false,
			use_geglu: true,
			c_type: "b8::E4m3",
			c_stride_expr: "b_rows / 2",
			scale_val: String::new(),
			scale_dscr: String::new(),
			head_dim: 0,
			sep_dim: 0,
			eps_val: String::new(),
			head_scale_val: String::new(),
			head_scale_dscr: String::new(),
			sep_scale_val: String::new(),
			sep_scale_dscr: String::new(),
			geglu_inp_scale_val: format_cpp_f64(self.inp_scale.value),
			geglu_inp_scale_dscr: self.inp_scale.description.as_ref().to_owned(),
			geglu_out_scale_val: format_cpp_f64(self.out_scale.value),
			geglu_out_scale_dscr: self.out_scale.description.as_ref().to_owned(),
			has_rrms_output: false,
		}
	}
}

fn format_cpp_f64(value: f64) -> String {
	format!("{value:.17e}")
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
pub struct NoExtraArgs<'a> {
	phantom: PhantomData<&'a ()>,
}

impl<'a> NoExtraArgs<'a> {
	pub fn new() -> Self {
		Self { phantom: PhantomData }
	}
}

pub struct RMSNormExtraArgs<'a> {
	pub rrms: &'a Tensor,
}

impl<'a> RMSNormExtraArgs<'a> {
	pub fn new(rrms: &'a Tensor) -> Self {
		Self { rrms }
	}
}

#[derive(Clone, Copy)]
pub struct GemmArgs<'a, Epilogue: GemmEpilogue> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
	pub c: &'a Tensor,
	pub extra: Epilogue::ExtraArgs<'a>,
}

//--------------------------------------------------------------------------------------------------

pub trait GemmLauncher<Epilogue: GemmEpilogue>: Sized {
	fn new(
		device: Rc<CudaDevice>,
		gemm_config: &GemmConfig,
		epilogue_config: &Epilogue::Config,
		cubin_path: &Path,
		config_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError>;

	fn launch(&self, args: GemmArgs<Epilogue>) -> Result<(), ErrPack<TensorOpError>>;
}

pub trait GemmEpilogue: Sized {
	type Config: EpilogueConfig;
	type Launcher: GemmLauncher<Self>;
	type ExtraArgs<'a>;
}

pub struct ScaleEpilogue;
pub struct RMSNormEpilogue;
pub struct ResidualEpilogue;
pub struct GeGluEpilogue;

impl GemmEpilogue for ScaleEpilogue {
	type Config = ScaleConfig;
	type Launcher = BasicGemmLauncher;
	type ExtraArgs<'a> = NoExtraArgs<'a>;
}

impl GemmEpilogue for RMSNormEpilogue {
	type Config = RMSNormConfig;
	type Launcher = RMSNormGemmLauncher;
	type ExtraArgs<'a> = RMSNormExtraArgs<'a>;
}

impl GemmEpilogue for ResidualEpilogue {
	type Config = ResidualConfig;
	type Launcher = BasicGemmLauncher;
	type ExtraArgs<'a> = NoExtraArgs<'a>;
}

impl GemmEpilogue for GeGluEpilogue {
	type Config = GeGluConfig;
	type Launcher = GeGluGemmLauncher;
	type ExtraArgs<'a> = NoExtraArgs<'a>;
}

//--------------------------------------------------------------------------------------------------

pub struct GemmKernel<Epilogue: GemmEpilogue> {
	pub name: String,
	pub dir_path: PathBuf,
	pub common_path: PathBuf,
	pub kernel_path: PathBuf,
	pub ptx_path: PathBuf,
	pub cubin_path: PathBuf,
	pub meta_path: PathBuf,
	pub meta_exe_path: PathBuf,
	pub meta_json_path: PathBuf,
	pub launcher: Epilogue::Launcher,
}

impl<Epilogue: GemmEpilogue> GemmKernel<Epilogue> {
	pub fn new(
		device: Rc<CudaDevice>,
		kernel_name: impl AsRef<str>,
		gemm_config: &GemmConfig,
		epilogue_config: &Epilogue::Config,
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
			diag.add_error(format!("invalid GEMM kernel name {kernel_name:?}"));
			return Err(KernelGeneratorError);
		}

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

		Self::generate_sources(
			gemm_config, epilogue_config,
			&common_path, &kernel_path, &meta_path,
			diag
		)?;
		Self::compile_kernel_ptx(&dir_path, &kernel_path, &ptx_path, diag)?;
		Self::compile_kernel_cubin(&dir_path, &ptx_path, &cubin_path, diag)?;
		Self::compile_meta_exe(&dir_path, &meta_path, &meta_exe_path, diag)?;
		Self::run_meta_exe(&dir_path, &meta_exe_path, &meta_json_path, diag)?;

		let launcher = Epilogue::Launcher::new(
			device, gemm_config, epilogue_config,
			&cubin_path, &meta_json_path, diag
		)?;

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

	pub fn launch<'a>(
		&'a self, args: GemmArgs<'a, Epilogue>
	) -> Result<(), ErrPack<TensorOpError>> {
		self.launcher.launch(args)
	}

	pub fn n_ops(&self, a: &Tensor, b: &Tensor) -> f64 {
		// TODO: This assumes that `b` is transposed and `a` is not
		#![allow(clippy::cast_precision_loss)]
		2.0 * (a.size(-2) as f64) * (a.size(-1) as f64) * (b.size(-2) as f64)
	}

	fn generate_sources(
		gemm_config: &GemmConfig,
		epilogue_config: &dyn EpilogueConfig,
		common_path: &Path,
		kernel_path: &Path,
		meta_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		if gemm_config.a.dtype != DType::Int8 {
			todo!("support GEMM inputs other than i8");
		}
		if gemm_config.b.dtype != DType::Int8 {
			todo!("support GEMM weights other than i8");
		}
		if !epilogue_config.is_c_dtype_allowed(gemm_config.c_dtype) {
			todo!("support GEMM output dtype incompatible with epilogue");
		}
		if gemm_config.a.trans {
			todo!("support transposed GEMM input A");
		}
		if !gemm_config.b.trans {
			todo!("support non-transposed GEMM input B");
		}
		if gemm_config.a.cols == 0 || gemm_config.b.cols == 0 {
			todo!("support empty GEMM dimensions");
		}

		let Some(b_rows) = gemm_config.b.rows else {
			todo!("support GEMM outputs with runtime column count");
		};
		let b_rows = Some(b_rows);
		let writer_template = epilogue_config.writer_template(b_rows);

		let common = BasicGemmCommonTemplate {
			a_cols: gemm_config.a.cols,
			b_rows,
			writer: &writer_template,
		};
		Self::write_generated_file(
			common_path,
			common.render().unwrap_or_else(|_| todo!("render GEMM common template")),
			diag,
		)?;

		let kernel = BasicGemmKernelTemplate {
			a_cols: gemm_config.a.cols,
			b_rows,
			writer: &writer_template,
		};
		Self::write_generated_file(
			kernel_path,
			kernel.render().unwrap_or_else(|_| todo!("render GEMM kernel template")),
			diag,
		)?;

		let meta = BasicGemmMetaTemplate {};
		Self::write_generated_file(
			meta_path,
			meta.render().unwrap_or_else(|_| todo!("render GEMM metadata template")),
			diag,
		)?;

		Ok(())
	}

	fn write_generated_file(
		path: &Path,
		source: String,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		match std::fs::write(path, source) {
			Ok(()) => Ok(()),
			Err(err) => {
				cold_path();
				diag.add_error(format!(
					"failed to write generated CUDA source {}: {err}",
					path.display(),
				));
				Err(KernelGeneratorError)
			},
		}
	}

	fn compile_kernel_ptx(
		_dir_path: &Path,
		kernel_path: &Path,
		ptx_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		let mut command = Self::nvcc_command("compute_86");
		command
			.arg("-ptx")
			.arg(kernel_path)
			.arg("-lineinfo")
			.arg("-o")
			.arg(ptx_path);

		Self::run_checked_command(
			&mut command,
			&format!(
				"nvcc failed while compiling {} to {}",
				kernel_path.display(),
				ptx_path.display(),
			),
			diag,
		)?;
		Ok(())
	}

	fn compile_kernel_cubin(
		_dir_path: &Path,
		ptx_path: &Path,
		cubin_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		let mut command = Self::nvcc_command("sm_86");
		command
			.arg("-Xptxas=-v")
			.arg("--cubin")
			.arg(ptx_path)
			.arg("-o")
			.arg(cubin_path);

		Self::run_checked_command(
			&mut command,
			&format!(
				"nvcc failed while compiling {} to {}",
				ptx_path.display(),
				cubin_path.display(),
			),
			diag,
		)?;
		Ok(())
	}

	fn compile_meta_exe(
		_dir_path: &Path,
		meta_path: &Path,
		meta_exe_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		let mut command = Self::nvcc_command("sm_86");
		command
			.arg(meta_path)
			.arg("-o")
			.arg(meta_exe_path);

		Self::run_checked_command(
			&mut command,
			&format!(
				"nvcc failed while compiling {} to {}",
				meta_path.display(),
				meta_exe_path.display(),
			),
			diag,
		)?;
		Ok(())
	}

	fn run_meta_exe(
		_dir_path: &Path,
		meta_exe_path: &Path,
		meta_json_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		let mut command = Command::new(meta_exe_path);
		let output = Self::run_checked_command(
			&mut command,
			&format!("failed to run GEMM metadata executable {}", meta_exe_path.display()),
			diag,
		)?;

		match std::fs::write(meta_json_path, output.stdout) {
			Ok(()) => Ok(()),
			Err(err) => {
				cold_path();
				diag.add_error(format!(
					"failed to write generated GEMM metadata {}: {err}",
					meta_json_path.display(),
				));
				Err(KernelGeneratorError)
			},
		}
	}

	fn nvcc_command(arch: &str) -> Command {
		const NVCC_PATH: &str = "/usr/local/cuda-12.6/bin/nvcc";

		let mut command = Command::new(NVCC_PATH);
		command
			.arg(format!("-arch={arch}"))
			.arg("-std=c++20")
			.arg("--ftz=true")
			.arg("--prec-div=false")
			.arg("--fmad=true")
			.arg("--use_fast_math")
			.arg("-I")
			.arg("/home/spock/prog/cutlass/tools/util/include/")
			.arg("-I")
			.arg("/home/spock/prog/cutlass/include/")
			.arg("-DX17_PRECISE_MATH=0")
			.arg("--expt-relaxed-constexpr")
			.arg("-maxrregcount=255")
			.arg("-O3");
		command
	}

	fn run_checked_command(
		command: &mut Command,
		context: &str,
		diag: &mut Diagnostics,
	) -> Result<Output, KernelGeneratorError> {
		let command_text = format!("{command:?}");
		let output = match command.output() {
			Ok(output) => output,
			Err(err) => {
				cold_path();
				diag.add_error(format!("{context}: failed to start {command_text}: {err}"));
				return Err(KernelGeneratorError);
			},
		};

		if !output.status.success() {
			cold_path();
			let stdout = String::from_utf8_lossy(&output.stdout);
			let stderr = String::from_utf8_lossy(&output.stderr);
			diag.add_error(format!(
				"{context} (status: {})\ncommand: {}\nstdout:\n{}\nstderr:\n{}",
				output.status,
				command_text,
				stdout,
				stderr,
			));
			return Err(KernelGeneratorError);
		}

		Ok(output)
	}
}

//--------------------------------------------------------------------------------------------------

// This struct should only contain items where the source of truth naturally is the C++ CUDA code.
// We don't want to mirror these values in Rust and so will transfer them via the meta.json file.
#[derive(Deserialize)]
pub struct GemmLauncherMetadata {
	pub THREADS_PER_BLOCK: usize,
	pub SMEM_BYTES: usize,
	pub M_PER_BLOCK: usize,
	pub N_PER_BLOCK: usize,
}

pub struct BasicGemmLauncher {
	pub a_cols: usize,
	pub b_rows: Option<NonZeroUsize>,

	pub a_dtype: DType,
	pub b_dtype: DType,
	pub c_dtype: DType,

	pub metadata: GemmLauncherMetadata,
	pub device: Rc<CudaDevice>,
	pub kernel: CudaKernel,
}

impl BasicGemmLauncher {
	pub fn new(
		device: Rc<CudaDevice>,
		gemm_config: &GemmConfig,
		cubin_path: &Path,
		config_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError> {
		let config = match std::fs::read_to_string(config_path) {
			Ok(s) => s,
			Err(err) => {
				cold_path();
				let file = config_path.to_string_lossy();
				diag.add_error(format!("Error reading file {file}: {err}"));
				return Err(KernelGeneratorError);
			},
		};

		let metadata = match serde_json::from_str::<GemmLauncherMetadata>(&config) {
			Ok(m) => m,
			Err(err) => {
				cold_path();
				let file = config_path.to_string_lossy();
				diag.add_error(format!("Failed to parse JSON data from {file}: {err}"));
				return Err(KernelGeneratorError);
			},
		};

		if metadata.THREADS_PER_BLOCK == 0
			|| metadata.M_PER_BLOCK == 0
			|| metadata.N_PER_BLOCK == 0
		{
			cold_path();
			let file = config_path.to_string_lossy();
			diag.add_error(format!("Invalid JSON metadata in {file}"));
			return Err(KernelGeneratorError);
		}

		let module = match device.stream.load_module_from_cubin(cubin_path) {
			Ok(module) => module,
			Err(err) => {
				cold_path();
				diag.add_error(format!(
					"failed to load generated GEMM CUDA module {}: {err}",
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
					"failed to load generated GEMM CUDA kernel \"kernel\" from {}: {err}",
					cubin_path.display()
				));
				return Err(KernelGeneratorError);
			},
		};

		Ok(Self {
			a_cols: gemm_config.a.cols,
			b_rows: gemm_config.b.rows,

			a_dtype: gemm_config.a.dtype,
			b_dtype: gemm_config.b.dtype,
			c_dtype: gemm_config.c_dtype,

			metadata,
			device,
			kernel,
		})
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
		if let Some(expected_rows) = expected_shape[0] && rows != expected_rows {
			cold_path();
			return Err(gemm_launch_error(format!(
				"GEMM input {tensor_name} has {rows} rows, but should have {expected_rows}"
			)));
		}
		if let Some(expected_cols) = expected_shape[1] && cols != expected_cols {
			cold_path();
			return Err(gemm_launch_error(format!(
				"GEMM input {tensor_name} has {cols} columns, but should have {expected_cols}"
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

	fn launch_gemm(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let [a_rows, _a_cols] = Self::matrix_shape(
			"a", a, [None, Some(self.a_cols)],
			&self.device, self.a_dtype
		)?;
		let [b_rows, _b_cols] = Self::matrix_shape(
			"b", b, [self.b_rows.map(NonZeroUsize::get), Some(self.a_cols)],
			&self.device, self.b_dtype
		)?;
		let [c_rows, c_cols] = Self::matrix_shape(
			"c", c, [Some(a_rows), Some(b_rows)],
			&self.device, self.c_dtype
		)?;

		// TODO - assume M_PER_BLOCK and N_PER_BLOCK are powers of 2
		if c_rows % self.metadata.M_PER_BLOCK != 0 || c_cols % self.metadata.N_PER_BLOCK != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"GEMM output shape [{c_rows}, {c_cols}] must be divisible by tile shape [{}, {}]",
				self.metadata.M_PER_BLOCK, self.metadata.N_PER_BLOCK
			)));
		}

		let grid_x = c_rows / self.metadata.M_PER_BLOCK;
		let grid_y = c_cols / self.metadata.N_PER_BLOCK;
		let mut a_ptr = unsafe { a.device_ptr().as_ptr::<c_void>() };
		let mut a_rows_arg = a_rows;
		let mut b_ptr = unsafe { b.device_ptr().as_ptr::<c_void>() };
		let mut b_rows_arg = b_rows;
		let mut c_ptr = unsafe { c.device_ptr().as_ptr::<c_void>() };

		let mut raw_args = [
			(&raw mut a_ptr).cast::<c_void>(),
			(&raw mut a_rows_arg).cast::<c_void>(),
			(&raw mut b_ptr).cast::<c_void>(),
			(&raw mut b_rows_arg).cast::<c_void>(),
			(&raw mut c_ptr).cast::<c_void>(),
		];
		unsafe {
			self.kernel.launch(
				&self.device.stream,
				[grid_x, grid_y, 1],
				[self.metadata.THREADS_PER_BLOCK, 1, 1],
				self.metadata.SMEM_BYTES,
				&mut raw_args,
			)
		}
	}
}

impl GemmLauncher<ScaleEpilogue> for BasicGemmLauncher {
	fn new(
		device: Rc<CudaDevice>,
		gemm_config: &GemmConfig,
		_epilogue_config: &ScaleConfig,
		cubin_path: &Path,
		config_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError> {
		Self::new(device, gemm_config, cubin_path, config_path, diag)
	}

	fn launch(&self, args: GemmArgs<ScaleEpilogue>) -> Result<(), ErrPack<TensorOpError>> {
		self.launch_gemm(args.a, args.b, args.c)
	}
}

impl GemmLauncher<ResidualEpilogue> for BasicGemmLauncher {
	fn new(
		device: Rc<CudaDevice>,
		gemm_config: &GemmConfig,
		_epilogue_config: &ResidualConfig,
		cubin_path: &Path,
		config_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError> {
		Self::new(device, gemm_config, cubin_path, config_path, diag)
	}

	fn launch(&self, args: GemmArgs<ResidualEpilogue>) -> Result<(), ErrPack<TensorOpError>> {
		self.launch_gemm(args.a, args.b, args.c)
	}
}

pub struct GeGluGemmLauncher {
	basic: BasicGemmLauncher,
}

impl GemmLauncher<GeGluEpilogue> for GeGluGemmLauncher {
	fn new(
		device: Rc<CudaDevice>,
		gemm_config: &GemmConfig,
		_epilogue_config: &GeGluConfig,
		cubin_path: &Path,
		config_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError> {
		let basic = BasicGemmLauncher::new(device, gemm_config, cubin_path, config_path, diag)?;
		Ok(Self { basic })
	}

	fn launch(&self, args: GemmArgs<GeGluEpilogue>) -> Result<(), ErrPack<TensorOpError>> {
		let basic = &self.basic;
		let GemmArgs {a, b, c, extra: _} = args;

		let [a_rows, _a_cols] = BasicGemmLauncher::matrix_shape(
			"a", a, [None, Some(basic.a_cols)],
			&basic.device, basic.a_dtype
		)?;
		let [b_rows, _b_cols] = BasicGemmLauncher::matrix_shape(
			"b", b, [basic.b_rows.map(NonZeroUsize::get), Some(basic.a_cols)],
			&basic.device, basic.b_dtype
		)?;
		if b_rows % 2 != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"GeGLU GEMM requires an even raw output column count, got {b_rows}"
			)));
		}
		let [c_rows, _c_cols] = BasicGemmLauncher::matrix_shape(
			"c", c, [Some(a_rows), Some(b_rows / 2)],
			&basic.device, basic.c_dtype
		)?;

		// TODO - assume M_PER_BLOCK and N_PER_BLOCK are powers of 2
		if c_rows % basic.metadata.M_PER_BLOCK != 0 || b_rows % basic.metadata.N_PER_BLOCK != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"GeGLU GEMM raw output shape [{c_rows}, {b_rows}] must be divisible by tile shape [{}, {}]",
				basic.metadata.M_PER_BLOCK, basic.metadata.N_PER_BLOCK
			)));
		}

		let grid_x = c_rows / basic.metadata.M_PER_BLOCK;
		let grid_y = b_rows / basic.metadata.N_PER_BLOCK;
		let mut a_ptr = unsafe { a.device_ptr().as_ptr::<c_void>() };
		let mut a_rows_arg = a_rows;
		let mut b_ptr = unsafe { b.device_ptr().as_ptr::<c_void>() };
		let mut b_rows_arg = b_rows;
		let mut c_ptr = unsafe { c.device_ptr().as_ptr::<c_void>() };

		let mut raw_args = [
			(&raw mut a_ptr).cast::<c_void>(),
			(&raw mut a_rows_arg).cast::<c_void>(),
			(&raw mut b_ptr).cast::<c_void>(),
			(&raw mut b_rows_arg).cast::<c_void>(),
			(&raw mut c_ptr).cast::<c_void>(),
		];
		unsafe {
			basic.kernel.launch(
				&basic.device.stream,
				[grid_x, grid_y, 1],
				[basic.metadata.THREADS_PER_BLOCK, 1, 1],
				basic.metadata.SMEM_BYTES,
				&mut raw_args,
			)
		}
	}
}

pub struct RMSNormGemmLauncher {
	basic: BasicGemmLauncher,
	head_dim: usize,
	sep_dim: usize,
}

impl GemmLauncher<RMSNormEpilogue> for RMSNormGemmLauncher {
	fn new(
		device: Rc<CudaDevice>,
		gemm_config: &GemmConfig,
		epilogue_config: &RMSNormConfig,
		cubin_path: &Path,
		config_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError> {
		let basic = BasicGemmLauncher::new(device, gemm_config, cubin_path, config_path, diag)?;
		Ok(Self {
			basic,
			head_dim: epilogue_config.head_dim,
			sep_dim: epilogue_config.sep_dim,
		})
	}

	fn launch(&self, args: GemmArgs<RMSNormEpilogue>) -> Result<(), ErrPack<TensorOpError>> {
		let Self {basic, head_dim, sep_dim} = &self;
		let GemmArgs {a, b, c, extra: RMSNormExtraArgs { rrms }} = args;

		let [a_rows, _a_cols] = BasicGemmLauncher::matrix_shape(
			"a", a, [None, Some(basic.a_cols)],
			&basic.device, basic.a_dtype
		)?;
		let [b_rows, _b_cols] = BasicGemmLauncher::matrix_shape(
			"b", b, [basic.b_rows.map(NonZeroUsize::get), Some(basic.a_cols)],
			&basic.device, basic.b_dtype
		)?;
		let [c_rows, c_cols] = BasicGemmLauncher::matrix_shape(
			"c", c, [Some(a_rows), Some(b_rows)],
			&basic.device, basic.c_dtype
		)?;

		// TODO - assume M_PER_BLOCK and N_PER_BLOCK are powers of 2
		if c_rows % basic.metadata.M_PER_BLOCK != 0 || c_cols % basic.metadata.N_PER_BLOCK != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"GEMM output shape [{c_rows}, {c_cols}] must be divisible by tile shape [{}, {}]",
				basic.metadata.M_PER_BLOCK, basic.metadata.N_PER_BLOCK
			)));
		}

		let chunk = head_dim + sep_dim;
		if chunk == 0 || b_rows % chunk != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"RMSNorm metadata dimensions [{head_dim}, {sep_dim}] do not divide b rows {b_rows}",
			)));
		}
		let rrms_cols = b_rows / chunk;
		BasicGemmLauncher::matrix_shape(
			"rrms", rrms, [Some(a_rows), Some(rrms_cols)],
			&basic.device, DType::F32
		)?;

		let grid_x = c_rows / basic.metadata.M_PER_BLOCK;
		let grid_y = c_cols / basic.metadata.N_PER_BLOCK;
		let mut a_ptr = unsafe { a.device_ptr().as_ptr::<c_void>() };
		let mut a_rows_arg = a_rows;
		let mut b_ptr = unsafe { b.device_ptr().as_ptr::<c_void>() };
		let mut b_rows_arg = b_rows;
		let mut c_ptr = unsafe { c.device_ptr().as_ptr::<c_void>() };
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
			basic.kernel.launch(
				&basic.device.stream,
				[grid_x, grid_y, 1],
				[basic.metadata.THREADS_PER_BLOCK, 1, 1],
				basic.metadata.SMEM_BYTES,
				&mut raw_args,
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

//--------------------------------------------------------------------------------------------------
