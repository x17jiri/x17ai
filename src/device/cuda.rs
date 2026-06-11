//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::num::NonZeroUsize;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::NonNull;
use std::rc::Rc;

use askama::Template;

use crate::dtype::DType;
use crate::{DeviceAllocError, ErrExtra, ErrPack, TensorOpError};

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

	pub fn new_named(device_id: usize, name: String) -> Result<Rc<Self>, ErrPack<TensorOpError>> {
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

#[derive(Template)]
#[template(escape = "none", path = "gemm_kernel.cu")]
struct GemmKernelTemplate<'a> {
	a_cols: usize,
	b_rows: Option<usize>,
	scale_val: String,
	scale_dscr: &'a str,
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
	let template = GemmKernelTemplate {
		a_cols: config.a.cols,
		b_rows,
		scale_val: format!("{scale_val:.17e}"),
		scale_dscr: scale.description.as_ref(),
	};

	template.render().unwrap_or_else(|_| todo!("render GEMM kernel template"))
}

//--------------------------------------------------------------------------------------------------

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

#[link(name = "dl")]
unsafe extern "C" {
	fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
	fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
	fn dlclose(handle: *mut c_void) -> c_int;
	fn dlerror() -> *const c_char;
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct GemmKernel {
	pub name: String,
	pub source_path: PathBuf,
	pub library_path: PathBuf,

	_library: DynamicLibrary,
	deinit: KernelLifecycleFn,
	launch: KernelLaunchFn,
}

impl GemmKernel {
	pub fn new(
		kernel_name: impl AsRef<str>,
		config: &GemmKernelConfig,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let kernel_name = kernel_name.as_ref();
		validate_kernel_name(kernel_name)?;

		let source = generate_gemm_kernel(config);
		let kernel_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache").join("kernels");
		let source_path = kernel_dir.join(format!("{kernel_name}.cu"));
		let library_path = kernel_dir.join(format!("{kernel_name}.so"));

		fs::create_dir_all(&kernel_dir).map_err(|err| {
			gemm_kernel_io_error(
				format!("failed to create kernel cache directory {}", kernel_dir.display()),
				err,
			)
		})?;
		fs::write(&source_path, source).map_err(|err| {
			gemm_kernel_io_error(
				format!("failed to write generated CUDA source {}", source_path.display()),
				err,
			)
		})?;

		compile_gemm_kernel(&source_path, &library_path)?;

		let library = DynamicLibrary::open(&library_path)?;
		let init = library.lifecycle_fn("x17ai_kernel_init")?;
		let deinit = library.lifecycle_fn("x17ai_kernel_deinit")?;
		let launch = library.launch_fn("x17ai_kernel_launch")?;
		call_kernel_lifecycle("x17ai_kernel_init", init)?;

		Ok(Self {
			name: kernel_name.to_owned(),
			source_path,
			library_path,
			_library: library,
			deinit,
			launch,
		})
	}

	/// # Safety
	///
	/// The device pointers must point to valid CUDA buffers with layouts expected by this kernel.
	pub unsafe fn run(
		&self,
		a: DevicePtr,
		a_rows: usize,
		b: DevicePtr,
		b_rows: usize,
		c: DevicePtr,
	) -> Result<(), ErrPack<TensorOpError>> {
		let err = unsafe {
			(self.launch)(
				a.as_ptr::<c_void>(), a_rows,
				b.as_ptr::<c_void>(), b_rows,
				c.as_ptr::<c_void>(),
			)
		};
		if err != 0 {
			return Err(gemm_kernel_error(format!(
				"x17ai_kernel_launch failed with CUDA error code {err}"
			)));
		}

		Ok(())
	}
}

impl Drop for GemmKernel {
	fn drop(&mut self) {
		let err = unsafe { (self.deinit)() };
		if err != 0 {
			log::error!("x17ai_kernel_deinit failed with CUDA error code {err}");
		}
	}
}

type KernelLifecycleFn = unsafe extern "C" fn() -> usize;
type KernelLaunchFn =
	unsafe extern "C" fn(*mut c_void, usize, *mut c_void, usize, *mut c_void) -> usize;

#[derive(Debug)]
struct DynamicLibrary {
	handle: NonNull<c_void>,
}

impl DynamicLibrary {
	fn open(path: &Path) -> Result<Self, ErrPack<TensorOpError>> {
		let path_cstr = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
			gemm_kernel_error(format!(
				"failed to load CUDA kernel library {}; path contains an interior NUL byte",
				path.display()
			))
		})?;

		clear_dlerror();
		let handle = unsafe { dlopen(path_cstr.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
		let Some(handle) = NonNull::new(handle) else {
			return Err(gemm_kernel_dl_error(format!(
				"failed to load CUDA kernel library {}",
				path.display()
			)));
		};

		Ok(Self { handle })
	}

	fn lifecycle_fn(&self, symbol_name: &str) -> Result<KernelLifecycleFn, ErrPack<TensorOpError>> {
		let symbol = self.symbol_ptr(symbol_name)?;
		Ok(unsafe { std::mem::transmute::<*mut c_void, KernelLifecycleFn>(symbol) })
	}

	fn launch_fn(&self, symbol_name: &str) -> Result<KernelLaunchFn, ErrPack<TensorOpError>> {
		let symbol = self.symbol_ptr(symbol_name)?;
		Ok(unsafe { std::mem::transmute::<*mut c_void, KernelLaunchFn>(symbol) })
	}

	fn symbol_ptr(&self, symbol_name: &str) -> Result<*mut c_void, ErrPack<TensorOpError>> {
		let symbol_cstr = CString::new(symbol_name).map_err(|_| {
			gemm_kernel_error(format!(
				"failed to load CUDA kernel symbol {symbol_name:?}; symbol contains an interior NUL byte"
			))
		})?;

		clear_dlerror();
		let symbol = unsafe { dlsym(self.handle.as_ptr(), symbol_cstr.as_ptr()) };
		if symbol.is_null() {
			return Err(gemm_kernel_dl_error(format!(
				"failed to load CUDA kernel symbol {symbol_name:?}"
			)));
		}

		Ok(symbol)
	}
}

impl Drop for DynamicLibrary {
	fn drop(&mut self) {
		unsafe {
			dlclose(self.handle.as_ptr());
		}
	}
}

fn validate_kernel_name(kernel_name: &str) -> Result<(), ErrPack<TensorOpError>> {
	if kernel_name.is_empty()
		|| kernel_name == "."
		|| kernel_name == ".."
		|| kernel_name.contains('/')
		|| kernel_name.contains('\\')
	{
		return Err(gemm_kernel_error(format!("invalid GEMM kernel name {kernel_name:?}")));
	}

	Ok(())
}

fn compile_gemm_kernel(
	source_path: &Path,
	library_path: &Path,
) -> Result<(), ErrPack<TensorOpError>> {
	const NVCC: &str = "/usr/local/cuda-12.6/bin/nvcc";

	let output = Command::new(NVCC)
		.current_dir(env!("CARGO_MANIFEST_DIR"))
		.arg("-arch=sm_86")
		.arg("-std=c++20")
		.arg("-Xptxas=-v")
		.arg("--ftz=true")
		.arg("--prec-div=true")
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
		.output()
		.map_err(|err| gemm_kernel_io_error("failed to run nvcc", err))?;

	if !output.status.success() {
		let stdout = String::from_utf8_lossy(&output.stdout);
		let stderr = String::from_utf8_lossy(&output.stderr);
		return Err(gemm_kernel_error(format!(
			"nvcc failed while compiling {} to {} (status: {})\nstdout:\n{}\nstderr:\n{}",
			source_path.display(),
			library_path.display(),
			output.status,
			stdout,
			stderr,
		)));
	}

	Ok(())
}

fn call_kernel_lifecycle(
	symbol_name: &str,
	symbol: KernelLifecycleFn,
) -> Result<(), ErrPack<TensorOpError>> {
	let err = unsafe { symbol() };
	if err != 0 {
		return Err(gemm_kernel_error(format!("{symbol_name} failed with CUDA error code {err}")));
	}
	Ok(())
}

fn gemm_kernel_error(message: impl Into<Cow<'static, str>>) -> ErrPack<TensorOpError> {
	ErrPack {
		code: TensorOpError::Device,
		extra: Some(Box::new(ErrExtra { message: message.into(), nested: None })),
	}
}

fn gemm_kernel_dl_error(message: impl Into<String>) -> ErrPack<TensorOpError> {
	let mut message = message.into();
	let detail = unsafe { dlerror_message() };
	if let Some(detail) = detail {
		message.push_str(": ");
		message.push_str(&detail);
	}

	gemm_kernel_error(message)
}

fn clear_dlerror() {
	unsafe {
		dlerror();
	}
}

unsafe fn dlerror_message() -> Option<String> {
	let err = unsafe { dlerror() };
	if err.is_null() {
		None
	} else {
		Some(unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned())
	}
}

//--------------------------------------------------------------------------------------------------
fn gemm_kernel_io_error(
	message: impl Into<Cow<'static, str>>,
	err: std::io::Error,
) -> ErrPack<TensorOpError> {
	ErrPack {
		code: TensorOpError::IOError,
		extra: Some(Box::new(ErrExtra {
			message: message.into(),
			nested: Some(Box::new(err)),
		})),
	}
}

//--------------------------------------------------------------------------------------------------
