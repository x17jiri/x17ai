//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::{borrow::Cow, fs, hint::cold_path, marker::PhantomData, num::NonZeroUsize, path::{Path, PathBuf}, process::{Command, Output}, rc::Rc};

use askama::Template;
use serde::Deserialize;

use crate::{ErrExtra, ErrPack, KernelGeneratorError, TensorOpError, device::cuda::{CudaDevice, Diagnostics, basic_gemm::{BasicGemmCommonTemplate, BasicGemmKernelTemplate, BasicGemmMetaTemplate, BasicGemmWriterTemplate}, cuda_shim::CudaKernel}, dtype::DType, tensor::Tensor};

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

pub enum EpilogueConfigEnum<'a> {
	Scale(&'a ScaleConfig),
	RMSNorm(&'a RMSNormConfig),
	Residual(&'a ResidualConfig),
	GeGlu(&'a GeGluConfig),
}

pub trait EpilogueConfig {
	fn to_config_enum<'a>(&'a self) -> EpilogueConfigEnum<'a>;
}

impl EpilogueConfig for ScaleConfig {
	fn to_config_enum<'a>(&'a self) -> EpilogueConfigEnum<'a> {
		EpilogueConfigEnum::Scale(self)
	}
}

impl EpilogueConfig for RMSNormConfig {
	fn to_config_enum<'a>(&'a self) -> EpilogueConfigEnum<'a> {
		EpilogueConfigEnum::RMSNorm(self)
	}
}

impl EpilogueConfig for ResidualConfig {
	fn to_config_enum<'a>(&'a self) -> EpilogueConfigEnum<'a> {
		EpilogueConfigEnum::Residual(self)
	}
}

impl EpilogueConfig for GeGluConfig {
	fn to_config_enum<'a>(&'a self) -> EpilogueConfigEnum<'a> {
		EpilogueConfigEnum::GeGlu(self)
	}
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
	rrms: &'a Tensor,
}

//--------------------------------------------------------------------------------------------------

pub trait GemmEpilogue {
	type Config: EpilogueConfig;
	type ExtraArgs<'a>;
}

pub struct ScaleEpilogue;
pub struct RMSNormEpilogue;
pub struct ResidualEpilogue;
pub struct GeGluEpilogue;

impl GemmEpilogue for ScaleEpilogue {
	type Config = ScaleConfig;
	type ExtraArgs<'a> = NoExtraArgs<'a>;
}

impl GemmEpilogue for RMSNormEpilogue {
	type Config = RMSNormConfig;
	type ExtraArgs<'a> = RMSNormExtraArgs<'a>;
}

impl GemmEpilogue for ResidualEpilogue {
	type Config = ResidualConfig;
	type ExtraArgs<'a> = NoExtraArgs<'a>;
}

impl GemmEpilogue for GeGluEpilogue {
	type Config = GeGluConfig;
	type ExtraArgs<'a> = NoExtraArgs<'a>;
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct GemmArgs<'a, Epilogue: GemmEpilogue> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
	pub c: &'a Tensor,
	pub extra: Epilogue::ExtraArgs<'a>,
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
	pub lanuncher: GemmLauncher<Epilogue>,
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
		match fs::create_dir_all(&dir_path) {
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
			gemm_config, epilogue_config.to_config_enum(),
			&common_path, &kernel_path, &meta_path,
			diag
		)?;
		Self::compile_kernel_ptx(&dir_path, &kernel_path, &ptx_path, diag)?;
		Self::compile_kernel_cubin(&dir_path, &ptx_path, &cubin_path, diag)?;
		Self::compile_meta_exe(&dir_path, &meta_path, &meta_exe_path, diag)?;
		Self::run_meta_exe(&dir_path, &meta_exe_path, &meta_json_path, diag)?;

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
			lanuncher: GemmLauncher::new(
				device, gemm_config, epilogue_config,
				&cubin_path, &meta_json_path, diag
			)?,
		})
	}

	pub fn launch<'a>(
		&'a self, args: GemmArgs<'a, Epilogue>
	) -> Result<(), ErrPack<TensorOpError>> {
		self.lanuncher.launch(args)
	}

	pub fn n_ops(&self, a: &Tensor, b: &Tensor) -> f64 {
		self.lanuncher.n_ops(a, b)
	}

	fn generate_sources(
		gemm_config: &GemmConfig,
		epilogue_config: EpilogueConfigEnum,
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
		match epilogue_config {
			EpilogueConfigEnum::GeGlu(_) if gemm_config.c_dtype == DType::E4m3 => {},
			EpilogueConfigEnum::GeGlu(_) => {
				todo!("support GeGLU GEMM outputs other than e4m3");
			},
			_ if gemm_config.c_dtype == DType::Int8 => {},
			_ => {
				todo!("support GEMM outputs other than i8");
			},
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
		let writer = Self::generate_writer_template(epilogue_config, b_rows);

		let common = BasicGemmCommonTemplate {
			a_cols: gemm_config.a.cols,
			b_rows,
			writer: &writer,
		};
		let kernel = BasicGemmKernelTemplate {
			a_cols: gemm_config.a.cols,
			b_rows,
			writer: &writer,
		};
		let meta = BasicGemmMetaTemplate {};

		Self::write_generated_file(
			common_path,
			common.render().unwrap_or_else(|_| todo!("render GEMM common template")),
			diag,
		)?;
		Self::write_generated_file(
			kernel_path,
			kernel.render().unwrap_or_else(|_| todo!("render GEMM kernel template")),
			diag,
		)?;
		Self::write_generated_file(
			meta_path,
			meta.render().unwrap_or_else(|_| todo!("render GEMM metadata template")),
			diag,
		)?;
		Ok(())
	}

	// TODO: We could add `fn writer_template(...)` to the `EpilogueConfig` trait
	// and get rid of this huge function
	#[allow(clippy::too_many_lines)]
	#[allow(clippy::needless_pass_by_value)]
	fn generate_writer_template<'e>(
		epilogue_config: EpilogueConfigEnum<'e>,
		b_rows: Option<NonZeroUsize>,
	) -> BasicGemmWriterTemplate {
		match epilogue_config {
			EpilogueConfigEnum::Scale(scale) => {
				if !scale.0.value.is_finite() {
					todo!("support non-finite GEMM scale values");
				}

				BasicGemmWriterTemplate {
					use_l2_norm: false,
					use_geglu: false,
					c_type: "b8::FixedI8",
					c_stride_expr: "b_rows",
					output_cols_divisor: 1,
					scale_val: Self::format_cpp_f64(scale.0.value),
					scale_dscr: scale.0.description.as_ref().to_owned(),
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
			},
			EpilogueConfigEnum::RMSNorm(rms_norm) => {
				if !rms_norm.eps.is_finite() || rms_norm.eps < 0.0 {
					todo!("support invalid RMSNorm eps values");
				}
				if rms_norm.head_dim == 0 || rms_norm.sep_dim == 0 {
					todo!("support empty RMSNorm GEMM epilogue dimensions");
				}
				if rms_norm.head_dim != rms_norm.sep_dim {
					todo!("support RMSNorm GEMM epilogues where head_dim != sep_dim");
				}
				if rms_norm.head_dim % 32 != 0 || rms_norm.sep_dim % 32 != 0 {
					todo!("support RMSNorm GEMM epilogue dimensions that are not multiples of 32");
				}
				if !rms_norm.head_scale.value.is_finite() || !rms_norm.sep_scale.value.is_finite() {
					todo!("support non-finite RMSNorm GEMM scale values");
				}
				let Some(b_rows) = b_rows else {
					todo!("support RMSNorm GEMM outputs with runtime column count");
				};
				let chunk = rms_norm.head_dim + rms_norm.sep_dim;
				if b_rows.get() % chunk != 0 {
					todo!("support RMSNorm GEMM output columns that are not divisible by head_dim + sep_dim");
				}

				#[allow(clippy::cast_precision_loss)]
				let head_scale = rms_norm.head_scale.value * f64::sqrt(rms_norm.head_dim as f64);
				BasicGemmWriterTemplate {
					use_l2_norm: true,
					use_geglu: false,
					c_type: "b8::FixedI8",
					c_stride_expr: "b_rows",
					output_cols_divisor: 1,
					scale_val: String::new(),
					scale_dscr: String::new(),
					head_dim: rms_norm.head_dim,
					sep_dim: rms_norm.sep_dim,
					eps_val: Self::format_cpp_f64(rms_norm.eps),
					head_scale_val: Self::format_cpp_f64(head_scale),
					head_scale_dscr: format!(
						"({}) * sqrt({})",
						rms_norm.head_scale.description.as_ref(),
						rms_norm.head_dim,
					),
					sep_scale_val: Self::format_cpp_f64(rms_norm.sep_scale.value),
					sep_scale_dscr: rms_norm.sep_scale.description.as_ref().to_owned(),
					geglu_inp_scale_val: String::new(),
					geglu_inp_scale_dscr: String::new(),
					geglu_out_scale_val: String::new(),
					geglu_out_scale_dscr: String::new(),
					has_rrms_output: true,
				}
			},
			EpilogueConfigEnum::Residual(_) => todo!("support residual GEMM epilogues"),
			EpilogueConfigEnum::GeGlu(geglu) => {
				if !geglu.inp_scale.value.is_finite() || geglu.inp_scale.value <= 0.0 {
					todo!("support invalid GeGLU input scale values");
				}
				if !geglu.out_scale.value.is_finite() || geglu.out_scale.value <= 0.0 {
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
					output_cols_divisor: 2,
					scale_val: String::new(),
					scale_dscr: String::new(),
					head_dim: 0,
					sep_dim: 0,
					eps_val: String::new(),
					head_scale_val: String::new(),
					head_scale_dscr: String::new(),
					sep_scale_val: String::new(),
					sep_scale_dscr: String::new(),
					geglu_inp_scale_val: Self::format_cpp_f64(geglu.inp_scale.value),
					geglu_inp_scale_dscr: geglu.inp_scale.description.as_ref().to_owned(),
					geglu_out_scale_val: Self::format_cpp_f64(geglu.out_scale.value),
					geglu_out_scale_dscr: geglu.out_scale.description.as_ref().to_owned(),
					has_rrms_output: false,
				}
			},
		}
	}

	fn format_cpp_f64(value: f64) -> String {
		format!("{value:.17e}")
	}

	fn write_generated_file(
		path: &Path,
		source: String,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		match fs::write(path, source) {
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

		match fs::write(meta_json_path, output.stdout) {
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

pub struct GemmLauncher<Epilogue: GemmEpilogue> {
	pub a_cols: usize,
	pub b_rows: Option<NonZeroUsize>,
	pub output_cols_divisor: usize,
	pub head_dim: usize,
	pub sep_dim: usize,
	pub has_rrms_output: bool,
	pub a_dtype: DType,
	pub b_dtype: DType,
	pub c_dtype: DType,

	pub metadata: GemmLauncherMetadata,
	pub device: Rc<CudaDevice>,
	pub kernel: CudaKernel,

	pub phantom: PhantomData<Epilogue>
}

impl<Epilogue: GemmEpilogue> GemmLauncher<Epilogue> {
	pub fn new(
		device: Rc<CudaDevice>,
		gemm_config: &GemmConfig,
		epilogue_config: &Epilogue::Config,
		cubin_path: &Path,
		config_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError> {
		let config = match std::fs::read_to_string(config_path) {
			Ok(s) => s,
			Err(err) => {
				cold_path();
				let file = config_path.to_string_lossy();
				diag.add_error(format!("Error reading file {file}"));
				return Err(KernelGeneratorError);
			},
		};

		let metadata = match serde_json::from_str::<GemmLauncherMetadata>(&config) {
			Ok(m) => m,
			Err(err) => {
				cold_path();
				let file = config_path.to_string_lossy();
				diag.add_error(format!("Failed to parse JSON data from {file}"));
				return Err(KernelGeneratorError);
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
			diag.add_error(format!("Invalid JSON metadata in {file}"));
			return Err(KernelGeneratorError);
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

	fn launch<'a>(
		&'a self, args: GemmArgs<'a, Epilogue>
	) -> Result<(), ErrPack<TensorOpError>> {
		let a = args.a;
		let b = args.b;
		let c = args.c;

		let [a_rows, _a_cols] = Self::matrix_shape(
			"a", a, [None, Some(self.launch_info.a_cols)],
			device, self.launch_info.a_dtype
		)?;
		let [b_rows, _b_cols] = Self::matrix_shape(
			"b", b, [self.launch_info.b_rows.map(NonZeroUsize::get), Some(self.launch_info.a_cols)],
			device, self.launch_info.b_dtype
		)?;
		let [c_rows, c_cols] = Self::matrix_shape(
			"c", c, [Some(a_rows), Some(b_rows / self.launch_info.output_cols_divisor)],
			device, self.launch_info.c_dtype
		)?;
		// TODO: `b_rows / self.launch_info.output_cols_divisor` silently truncates if `b_rows`
		// is not divisible by the divisor.
		// Then `c_raw_cols` is reconstructed from the truncated value.

		// TODO - assume M_PER_BLOCK and N_PER_BLOCK are powers of 2
		let c_raw_cols = c_cols * self.launch_info.output_cols_divisor;
		if c_rows % self.metadata.M_PER_BLOCK != 0 || c_raw_cols % self.metadata.N_PER_BLOCK != 0 {
			cold_path();
			return Err(gemm_launch_error(format!(
				"GEMM output shape [{c_rows}, {c_raw_cols}] must be divisible by tile shape [{}, {}]",
				self.metadata.M_PER_BLOCK, self.metadata.N_PER_BLOCK
			)));
		}

		let grid_x = c_rows / self.metadata.M_PER_BLOCK;
		let grid_y = c_raw_cols / self.metadata.N_PER_BLOCK;
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

impl GemmLauncher {
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

//--------------------------------------------------------------------------------------------------
