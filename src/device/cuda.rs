//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::ffi::c_void;
use std::fs;
use std::hint::cold_path;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::NonNull;
use std::rc::Rc;

use askama::Template;

use crate::device::cuda_shim::CudaStreamHandle;
use crate::dtype::DType;
use crate::util::dyn_loader::DynamicLibrary;
use crate::{DeviceAllocError, ErrExtra, ErrPack, KernelGeneratorError, TensorOpError};

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

type KernelInitFn = unsafe extern "C" fn(*mut c_void) -> usize;

type KernelDeinitFn = unsafe extern "C" fn() -> usize;

type KernelLaunchFn = unsafe extern "C" fn(
	*mut CudaStreamHandle,
	*mut c_void, usize,
	*mut c_void, usize,
	*mut c_void
) -> usize;

pub struct Diagnostic {
	pub is_error: bool,
	pub message: String,
}

pub struct Diagnostics {
	pub list: Vec<Diagnostic>,
	pub err_count: usize,
}

impl Diagnostics {
	pub fn new() -> Diagnostics {
		Self { list: Vec::new(), err_count: 0 }
	}

	pub fn add_error(&mut self, message: String) {
		self.err_count += 1;
		self.list.push(Diagnostic { is_error: true, message });
	}
}

#[derive(Debug)]
pub struct GemmKernel {
	pub name: String,
	pub source_path: PathBuf,
	pub library_path: PathBuf,

	_library: DynamicLibrary,
	launch_fn: KernelLaunchFn,
	deinit_fn: KernelDeinitFn,
}

impl GemmKernel {
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

		let err = unsafe { init_fn(device.stream.cuda_context()) };
		if err != 0 {
			cold_path();
			diag.add_error(format!("x17ai_kernel_init failed with CUDA error code {err}"));
			return Err(KernelGeneratorError);
		}

		Ok(Self {
			name: kernel_name.to_owned(),
			source_path,
			library_path,
			_library: library,
			deinit_fn,
			launch_fn,
		})
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
		let err = unsafe {
			(self.launch_fn)(
				device.stream.handle(),
				a.as_ptr::<c_void>(), a_rows,
				b.as_ptr::<c_void>(), b_rows,
				c.as_ptr::<c_void>(),
			)
		};
		if err != 0 {
			cold_path();
			return Err(ErrPack {
				code: TensorOpError::Device,
				extra: Some(Box::new(ErrExtra {
					message: format!("x17ai_kernel_launch failed with CUDA error code {err}").into(),
					nested: None,
				})),
			});
		}
		Ok(())
	}
}

impl Drop for GemmKernel {
	fn drop(&mut self) {
		let err = unsafe { (self.deinit_fn)() };
		if err != 0 {
			log::error!("x17ai_kernel_deinit failed with CUDA error code {err}");
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
