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
use crate::device::cuda::basic_gemm::{BasicGemmCommonTemplate, BasicGemmKernelLauncher, BasicGemmKernelTemplate, BasicGemmMetaTemplate};
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::{DeviceAllocError, ErrPack, KernelGeneratorError, TensorOpError};

use super::{Device, DevicePtr};

pub mod cuda_shim;
pub mod basic_gemm;

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

pub struct RMSNormEpulogue {
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
	RMSNorm(RMSNormEpulogue),
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

pub struct GemmSources {
	pub common: String,
	pub kernel: String,
	pub meta: String,
}

pub trait GemmKernelLauncher {
	fn launch(
		&self, device: &CudaDevice, a: &Tensor, b: &Tensor, c: &Tensor
	) -> Result<(), ErrPack<TensorOpError>>;
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

		let sources = Self::generate_sources(config);
		let dir_path = Path::new("cache").join("kernels").join(kernel_name);
		let common_path = dir_path.join("common.cuh");
		let kernel_path = dir_path.join("kernel.cu");
		let ptx_path = dir_path.join("kernel.ptx");
		let cubin_path = dir_path.join("kernel.cubin");
		let meta_path = dir_path.join("meta.cu");
		let meta_exe_path = dir_path.join("meta");
		let meta_json_path = dir_path.join("meta.json");

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

		Self::write_generated_file(&common_path, sources.common, diag)?;
		Self::write_generated_file(&kernel_path, sources.kernel, diag)?;
		Self::write_generated_file(&meta_path, sources.meta, diag)?;

		Self::compile_kernel_ptx(&dir_path, diag)?;
		Self::compile_kernel_cubin(&dir_path, diag)?;
		Self::compile_meta_exe(&dir_path, diag)?;
		Self::run_meta_exe(&dir_path, &meta_json_path, diag)?;
		let launcher = match BasicGemmKernelLauncher::new(
			device.as_ref(),
			&cubin_path,
			&meta_json_path
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
	pub fn run(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.launcher.launch(&self.device, a, b, c)
	}

	fn generate_sources(config: &GemmKernelConfig) -> GemmSources {
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

		let scale = match &config.epilogue {
			GemmEpilogue::Scale(scale) => scale,
			_ => todo!("support GEMM epilogues other than scale"),
		};
		if !scale.value.is_finite() {
			todo!("support non-finite GEMM scale values");
		}

		let Some(b_rows) = config.b.rows.map(std::num::NonZeroUsize::get) else {
			todo!("support GEMM outputs with runtime column count");
		};
		let b_rows = Some(b_rows);
		let scale_val = scale.value;

		let common = BasicGemmCommonTemplate {
			a_cols: config.a.cols,
			b_rows,
			scale_val: format!("{scale_val:.17e}"),
			scale_dscr: scale.description.as_ref(),
		};
		let kernel = BasicGemmKernelTemplate {
			b_rows,
		};
		let meta = BasicGemmMetaTemplate;

		GemmSources {
			common: common.render().unwrap_or_else(|_| todo!("render GEMM common template")),
			kernel: kernel.render().unwrap_or_else(|_| todo!("render GEMM kernel template")),
			meta: meta.render().unwrap_or_else(|_| todo!("render GEMM metadata template")),
		}
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
		dir_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		let mut command = Self::nvcc_command("compute_86");
		command
			.current_dir(dir_path)
			.arg("-ptx")
			.arg("kernel.cu")
			.arg("-lineinfo")
			.arg("-o")
			.arg("kernel.ptx");

		Self::run_checked_command(
			&mut command,
			"nvcc failed while compiling kernel.cu to kernel.ptx",
			diag,
		)?;
		Ok(())
	}

	fn compile_kernel_cubin(
		dir_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		let mut command = Self::nvcc_command("sm_86");
		command
			.current_dir(dir_path)
			.arg("-Xptxas=-v")
			.arg("--cubin")
			.arg("kernel.ptx")
			.arg("-o")
			.arg("kernel.cubin");

		Self::run_checked_command(
			&mut command,
			"nvcc failed while compiling kernel.ptx to kernel.cubin",
			diag,
		)?;
		Ok(())
	}

	fn compile_meta_exe(
		dir_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		let mut command = Self::nvcc_command("sm_86");
		command
			.current_dir(dir_path)
			.arg("meta.cu")
			.arg("-o")
			.arg("meta");

		Self::run_checked_command(
			&mut command,
			"nvcc failed while compiling meta.cu to meta",
			diag,
		)?;
		Ok(())
	}

	fn run_meta_exe(
		dir_path: &Path,
		meta_json_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<(), KernelGeneratorError> {
		let mut command = Command::new("./meta");
		command.current_dir(dir_path);
		let output = Self::run_checked_command(
			&mut command,
			"failed to run GEMM metadata executable",
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

/*
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

	let scale = match &config.epilogue {
		GemmEpilogue::Scale(scale) => scale,
		_ => todo!("support GEMM epilogues other than scale"),
	};
	if !scale.value.is_finite() {
		todo!("support non-finite GEMM scale values");
	}

	let scale_val = scale.value;
	let Some(b_rows_val) = config.b.rows.map(NonZeroUsize::get) else {
		todo!("support GEMM outputs with runtime column count");
	};
	let b_rows = Some(b_rows_val);
	let template = basic_gemm::BasicGemmCommonTemplate {
		a_cols: config.a.cols,
		b_rows,
		scale_val: format!("{scale_val:.17e}"),
		scale_dscr: scale.description.as_ref(),
	};

	template.render().unwrap_or_else(|_| todo!("render GEMM kernel template"))
}

//--------------------------------------------------------------------------------------------------

type KernelInitFn = unsafe extern "C" fn(*mut c_void) -> *mut DiagnosticBuffer;

type KernelDeinitFn = unsafe extern "C" fn() -> *mut DiagnosticBuffer;

type KernelLaunchFn = unsafe extern "C" fn(
	*mut CudaStreamHandle,
	*mut c_void, usize,
	*mut c_void, usize,
	*mut c_void
) -> *mut DiagnosticBuffer;

#[derive(Debug)]
pub struct GemmKernel_old {
	pub name: String,
	pub source_path: PathBuf,
	pub library_path: PathBuf,

	a_cols: usize,
	_library: DynamicLibrary,
	launch_fn: KernelLaunchFn,
	deinit_fn: KernelDeinitFn,
}

impl GemmKernel_old {
	pub fn new(
		device: &CudaDevice,
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

		let source = generate_gemm_kernel(config);

		const nvcc_path: &str = "/usr/local/cuda-12.6/bin/nvcc";
		let working_dir = Path::new(".");

		let kernel_dir = working_dir.join("cache").join("kernels");
		let source_path = kernel_dir.join(format!("{kernel_name}.cu"));
		let library_path = kernel_dir.join(format!("{kernel_name}.so"));

		match fs::create_dir_all(&kernel_dir) {
			Ok(()) => {},
			Err(err) => {
				cold_path();
				let kernel_dir = kernel_dir.display();
				diag.add_error(format!(
					"failed to create kernel cache directory {kernel_dir}: {err}"
				));
				return Err(KernelGeneratorError);
			},
		};
		match fs::write(&source_path, source) {
			Ok(()) => {},
			Err(err) => {
				cold_path();
				let source_path = source_path.display();
				diag.add_error(format!(
					"failed to write generated CUDA source {source_path}: {err}"
				));
				return Err(KernelGeneratorError);
			},
		};

		compile_gemm_kernel(nvcc_path, &working_dir, &source_path, &library_path, diag)?;

		let library = match DynamicLibrary::open(&library_path) {
			Ok(lib) => lib,
			Err(msg) => {
				cold_path();
				diag.add_error(msg);
				return Err(KernelGeneratorError);
			},
		};
		let init_fn = match library.get_symbol("x17ai_kernel_init") {
			Ok(sym) => unsafe { std::mem::transmute::<*mut c_void, KernelInitFn>(sym) },
			Err(msg) => {
				cold_path();
				diag.add_error(msg);
				return Err(KernelGeneratorError);
			},
		};
		let deinit_fn = match library.get_symbol("x17ai_kernel_deinit") {
			Ok(sym) => unsafe { std::mem::transmute::<*mut c_void, KernelDeinitFn>(sym) },
			Err(msg) => {
				cold_path();
				diag.add_error(msg);
				return Err(KernelGeneratorError);
			},
		};
		let launch_fn = match library.get_symbol("x17ai_kernel_launch") {
			Ok(sym) => unsafe { std::mem::transmute::<*mut c_void, KernelLaunchFn>(sym) },
			Err(msg) => {
				cold_path();
				diag.add_error(msg);
				return Err(KernelGeneratorError);
			},
		};

		let diagnostic = unsafe { init_fn(device.stream.cuda_context()) };
		if !diagnostic.is_null() {
			cold_path();
			diag.add_error(diagnostic_to_string(
				diagnostic,
				"x17ai_kernel_init failed without diagnostic",
			));
			return Err(KernelGeneratorError);
		}

		Ok(Self {
			name: kernel_name.to_owned(),
			source_path,
			library_path,
			a_cols: config.a.cols,
			_library: library,
			deinit_fn,
			launch_fn,
		})
	}

	pub fn n_ops(&self, a_rows: usize, b_rows: usize) -> usize {
		a_rows
			.checked_mul(b_rows)
			.and_then(|v| v.checked_mul(self.a_cols))
			.and_then(|v| v.checked_mul(2))
			.expect("GEMM operation count overflow")
	}

	/// # Safety
	///
	/// The device pointers must point to valid CUDA buffers with layouts expected by this kernel.
	pub unsafe fn run(
		&self,
		device: &CudaDevice,
		a: DevicePtr, a_rows: usize,
		b: DevicePtr, b_rows: usize,
		c: DevicePtr,
	) -> Result<(), ErrPack<TensorOpError>> {
		let diagnostic = unsafe {
			(self.launch_fn)(
				device.stream.handle(),
				a.as_ptr::<c_void>(), a_rows,
				b.as_ptr::<c_void>(), b_rows,
				c.as_ptr::<c_void>(),
			)
		};
		if !diagnostic.is_null() {
			cold_path();
			return Err(ErrPack {
				code: TensorOpError::Device,
				extra: Some(Box::new(ErrExtra {
					message: diagnostic_to_string(
						diagnostic,
						"x17ai_kernel_launch failed without diagnostic",
					).into(),
					nested: None,
				})),
			});
		}
		Ok(())
	}
}

impl Drop for GemmKernel_old {
	fn drop(&mut self) {
		let diagnostic = unsafe { (self.deinit_fn)() };
		if !diagnostic.is_null() {
			log::error!(
				"{}",
				diagnostic_to_string(diagnostic, "x17ai_kernel_deinit failed without diagnostic")
			);
		}
	}
}

fn compile_gemm_kernel(
	nvcc_path: &str,
	working_dir: &Path,
	source_path: &Path,
	library_path: &Path,
	diag: &mut Diagnostics,
) -> Result<(), KernelGeneratorError> {
	let output = Command::new(nvcc_path)
		.current_dir(working_dir)
		.arg("-arch=sm_86")
		.arg("-std=c++20")
		.arg("-Xptxas=-v")
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
		.arg("-O3")
		.arg("-shared")
		.arg("-Xcompiler")
		.arg("-fPIC")
		.arg("-cudart=shared")
		.arg(source_path)
		.arg("-lineinfo")
		.arg("-o")
		.arg(library_path)
		.output();
	let output = match output {
		Ok(o) => o,
		Err(err) => {
			cold_path();
			diag.add_error(format!("failed to run nvcc: {err}"));
			return Err(KernelGeneratorError);
		},
	};

	if !output.status.success() {
		let stdout = String::from_utf8_lossy(&output.stdout);
		let stderr = String::from_utf8_lossy(&output.stderr);
		diag.add_error(format!(
			"nvcc failed while compiling {} to {} (status: {})\nstdout:\n{}\nstderr:\n{}",
			source_path.display(),
			library_path.display(),
			output.status,
			stdout,
			stderr,
		));
		return Err(KernelGeneratorError);
	}

	Ok(())
}

//--------------------------------------------------------------------------------------------------
*/
