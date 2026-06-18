//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::fs;
use std::hint::cold_path;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::ptr::NonNull;
use std::rc::Rc;

use askama::Template;

use cuda_shim::{CudaEventTimer, CudaStream};
use crate::device::cuda::basic_gemm::{
	BasicGemmCommonTemplate, BasicGemmKernelLauncher, BasicGemmKernelTemplate,
	BasicGemmLaunchInfo, BasicGemmMetaTemplate, BasicGemmWriterTemplate,
};
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::{DeviceAllocError, ErrPack, KernelGeneratorError, TensorOpError};

use super::{Device, DevicePtr};

pub mod cuda_shim;
pub mod basic_gemm;
pub mod gemm;

//--------------------------------------------------------------------------------------------------

pub struct CudaDevice {
	name: String,
	stream: CudaStream,
}

impl CudaDevice {
	pub fn new(device_id: usize) -> Result<Rc<Self>, ErrPack<TensorOpError>> {
		Self::new_named(device_id, format!("CUDA:{device_id}"))
	}

	pub fn new_named(device_id: usize, name: String) -> Result<Rc<Self>, ErrPack<TensorOpError>> {
		let stream = CudaStream::new(device_id)?;
		Ok(Rc::new(Self { name, stream }))
	}

	pub fn synchronize(&self) -> Result<(), ErrPack<TensorOpError>> {
		self.stream.synchronize()
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

pub struct CudaTimer<'a> {
	_device: &'a CudaDevice,
	timer: CudaEventTimer,
}

impl<'a> CudaTimer<'a> {
	pub fn new(device: &'a CudaDevice) -> Result<Self, ErrPack<TensorOpError>> {
		let timer = CudaEventTimer::new(&device.stream)?;
		Ok(Self { _device: device, timer })
	}

	pub fn start(&self) -> Result<(), ErrPack<TensorOpError>> {
		self.timer.start()
	}

	pub fn stop(&self) -> Result<(), ErrPack<TensorOpError>> {
		self.timer.stop()
	}

	pub fn elapsed_seconds(&self) -> Result<f64, ErrPack<TensorOpError>> {
		self.timer.elapsed_seconds()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Scale {
	pub value: f64,
	pub description: Cow<'static, str>,
}

pub struct RMSNormEpilogue {
	pub eps: f64,
	pub head_dim: usize,
	pub sep_dim: usize,

	pub head_scale: Scale,
	pub sep_scale: Scale,
}

pub struct ResidualEpilogue {
	pub old_scale: Scale,
	pub new_scale: Scale,
	pub out_scale: Scale,
}

pub struct GeGluEpilogue {
	pub inp_scale: Scale,
	pub out_scale: Scale,
}

pub enum GemmEpilogue {
	Scale(Scale),
	RMSNorm(RMSNormEpilogue),
	Residual(ResidualEpilogue),
	GeGlu(GeGluEpilogue),
}

pub struct GemmInput {
	pub dtype: DType,
	pub cols: usize,
	pub rows: Option<NonZeroUsize>,
	pub trans: bool,
}

pub struct GemmKernelConfig {
	pub a: GemmInput,
	pub b: GemmInput,

	pub epilogue: GemmEpilogue,
	pub c_dtype: DType,
}

pub struct Diagnostic {
	pub is_error: bool,
	pub message: String,
}

pub struct Diagnostics {
	pub list: Vec<Diagnostic>,
	pub err_count: usize,
}

impl Diagnostics {
	pub fn new() -> Self {
		Self { list: Vec::new(), err_count: 0 }
	}

	pub fn add_error(&mut self, message: String) {
		self.err_count += 1;
		self.list.push(Diagnostic { is_error: true, message });
	}
}

impl Default for Diagnostics {
	fn default() -> Self {
		Self::new()
	}
}

#[derive(Clone, Copy)]
pub enum GemmKernelExtraArgs<'a> {
	None,
	RMSNorm {
		rrms: &'a Tensor,
	},
}

#[derive(Clone, Copy)]
pub struct GemmKernelArgs<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
	pub c: &'a Tensor,
	pub extra: GemmKernelExtraArgs<'a>,
}

pub trait GemmKernelLauncher {
	fn launch(
		&self, device: &CudaDevice, args: GemmKernelArgs<'_>
	) -> Result<(), ErrPack<TensorOpError>>;

	fn n_ops(&self, a: &Tensor, b: &Tensor) -> f64;
}

pub struct GemmKernel {
	pub name: String,
	pub dir_path: PathBuf,
	pub common_path: PathBuf,
	pub kernel_path: PathBuf,
	pub ptx_path: PathBuf,
	pub cubin_path: PathBuf,
	pub meta_path: PathBuf,
	pub meta_exe_path: PathBuf,
	pub meta_json_path: PathBuf,
	pub device: Rc<CudaDevice>,
	pub launcher: Box<dyn GemmKernelLauncher>
}

impl GemmKernel {
	pub fn new(
		device: Rc<CudaDevice>,
		kernel_name: impl AsRef<str>,
		config: &GemmKernelConfig,
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

		Self::generate_sources(config, &common_path, &kernel_path, &meta_path, diag)?;
		Self::compile_kernel_ptx(&dir_path, &kernel_path, &ptx_path, diag)?;
		Self::compile_kernel_cubin(&dir_path, &ptx_path, &cubin_path, diag)?;
		Self::compile_meta_exe(&dir_path, &meta_path, &meta_exe_path, diag)?;
		Self::run_meta_exe(&dir_path, &meta_exe_path, &meta_json_path, diag)?;
		let launch_info = Self::generate_launch_info(config);
		let launcher = match BasicGemmKernelLauncher::new(
			device.as_ref(),
			&cubin_path,
			&meta_json_path,
			launch_info,
		) {
			Ok(launcher) => launcher,
			Err(err) => {
				cold_path();
				diag.add_error(format!("failed to load generated GEMM kernel: {err}"));
				return Err(KernelGeneratorError);
			},
		};

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
			device,
			launcher: Box::new(launcher),
		})
	}

	#[inline]
	pub fn run(&self, args: GemmKernelArgs<'_>) -> Result<(), ErrPack<TensorOpError>> {
		self.launcher.launch(&self.device, args)
	}

	#[inline]
	pub fn n_ops(&self, a: &Tensor, b: &Tensor) -> f64 {
		self.launcher.n_ops(a, b)
	}

	fn generate_sources(
		config: &GemmKernelConfig,
		common_path: &Path,
		kernel_path: &Path,
		meta_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		if config.a.dtype != DType::Int8 {
			todo!("support GEMM inputs other than i8");
		}
		if config.b.dtype != DType::Int8 {
			todo!("support GEMM weights other than i8");
		}
		match &config.epilogue {
			GemmEpilogue::GeGlu(_) if config.c_dtype == DType::E4m3 => {},
			GemmEpilogue::GeGlu(_) => {
				todo!("support GeGLU GEMM outputs other than e4m3");
			},
			_ if config.c_dtype == DType::Int8 => {},
			_ => {
				todo!("support GEMM outputs other than i8");
			},
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

		let Some(b_rows) = config.b.rows else {
			todo!("support GEMM outputs with runtime column count");
		};
		let b_rows = Some(b_rows);
		let writer = Self::generate_writer_template(config, b_rows);

		let common = BasicGemmCommonTemplate {
			a_cols: config.a.cols,
			b_rows,
			writer: &writer,
		};
		let kernel = BasicGemmKernelTemplate {
			a_cols: config.a.cols,
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

	fn generate_launch_info(config: &GemmKernelConfig) -> BasicGemmLaunchInfo {
		let writer = Self::generate_writer_template(config, config.b.rows);
		BasicGemmLaunchInfo {
			a_cols: config.a.cols,
			b_rows: config.b.rows,
			output_cols_divisor: writer.output_cols_divisor,
			head_dim: writer.head_dim,
			sep_dim: writer.sep_dim,
			has_rrms_output: writer.has_rrms_output,
			a_dtype: config.a.dtype,
			b_dtype: config.b.dtype,
			c_dtype: config.c_dtype,
		}
	}

	fn generate_writer_template(
		config: &GemmKernelConfig,
		b_rows: Option<NonZeroUsize>,
	) -> BasicGemmWriterTemplate {
		match &config.epilogue {
			GemmEpilogue::Scale(scale) => {
				if !scale.value.is_finite() {
					todo!("support non-finite GEMM scale values");
				}

				BasicGemmWriterTemplate {
					use_l2_norm: false,
					use_geglu: false,
					c_type: "b8::FixedI8",
					c_stride_expr: "b_rows",
					output_cols_divisor: 1,
					scale_val: Self::format_cpp_f64(scale.value),
					scale_dscr: scale.description.as_ref().to_owned(),
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
			GemmEpilogue::RMSNorm(epilogue) => {
				if !epilogue.eps.is_finite() || epilogue.eps < 0.0 {
					todo!("support invalid RMSNorm eps values");
				}
				if epilogue.head_dim == 0 || epilogue.sep_dim == 0 {
					todo!("support empty RMSNorm GEMM epilogue dimensions");
				}
				if epilogue.head_dim != epilogue.sep_dim {
					todo!("support RMSNorm GEMM epilogues where head_dim != sep_dim");
				}
				if epilogue.head_dim % 32 != 0 || epilogue.sep_dim % 32 != 0 {
					todo!("support RMSNorm GEMM epilogue dimensions that are not multiples of 32");
				}
				if !epilogue.head_scale.value.is_finite() || !epilogue.sep_scale.value.is_finite() {
					todo!("support non-finite RMSNorm GEMM scale values");
				}
				let Some(b_rows) = b_rows else {
					todo!("support RMSNorm GEMM outputs with runtime column count");
				};
				let chunk = epilogue.head_dim + epilogue.sep_dim;
				if b_rows.get() % chunk != 0 {
					todo!("support RMSNorm GEMM output columns that are not divisible by head_dim + sep_dim");
				}

				let head_scale = epilogue.head_scale.value * f64::sqrt(epilogue.head_dim as f64);
				BasicGemmWriterTemplate {
					use_l2_norm: true,
					use_geglu: false,
					c_type: "b8::FixedI8",
					c_stride_expr: "b_rows",
					output_cols_divisor: 1,
					scale_val: String::new(),
					scale_dscr: String::new(),
					head_dim: epilogue.head_dim,
					sep_dim: epilogue.sep_dim,
					eps_val: Self::format_cpp_f64(epilogue.eps),
					head_scale_val: Self::format_cpp_f64(head_scale),
					head_scale_dscr: format!(
						"({}) * sqrt({})",
						epilogue.head_scale.description.as_ref(),
						epilogue.head_dim,
					),
					sep_scale_val: Self::format_cpp_f64(epilogue.sep_scale.value),
					sep_scale_dscr: epilogue.sep_scale.description.as_ref().to_owned(),
					geglu_inp_scale_val: String::new(),
					geglu_inp_scale_dscr: String::new(),
					geglu_out_scale_val: String::new(),
					geglu_out_scale_dscr: String::new(),
					has_rrms_output: true,
				}
			},
			GemmEpilogue::Residual(_) => todo!("support residual GEMM epilogues"),
			GemmEpilogue::GeGlu(epilogue) => {
				if !epilogue.inp_scale.value.is_finite() || epilogue.inp_scale.value <= 0.0 {
					todo!("support invalid GeGLU input scale values");
				}
				if !epilogue.out_scale.value.is_finite() || epilogue.out_scale.value <= 0.0 {
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
					geglu_inp_scale_val: Self::format_cpp_f64(epilogue.inp_scale.value),
					geglu_inp_scale_dscr: epilogue.inp_scale.description.as_ref().to_owned(),
					geglu_out_scale_val: Self::format_cpp_f64(epilogue.out_scale.value),
					geglu_out_scale_dscr: epilogue.out_scale.description.as_ref().to_owned(),
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
