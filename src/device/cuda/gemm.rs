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
use std::rc::Rc;

use askama::Template;
use serde::Deserialize;

use crate::device::cuda::gemm_templates::{BasicGemmCommonTemplate, BasicGemmKernelTemplate, BasicGemmMetaTemplate, BasicGemmWriterTemplate};
use crate::{Diagnostics, ErrPack, KernelGeneratorError, TensorOpError};
use crate::dtype::DType;
use crate::tensor::Tensor;

use super::{CudaDevice, cuda_shim::CudaKernel, kernel_build, tensor_check};

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

	fn writer_template(&self, c_cols: Option<NonZeroUsize>) -> BasicGemmWriterTemplate;
}

impl EpilogueConfig for ScaleConfig {
	fn is_c_dtype_allowed(&self, c_dtype: DType) -> bool {
		c_dtype == DType::Int8
	}

	fn writer_template(&self, _c_cols: Option<NonZeroUsize>) -> BasicGemmWriterTemplate {
		if !self.0.value.is_finite() {
			todo!("error for invalid values");
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

	fn writer_template(&self, _c_cols: Option<NonZeroUsize>) -> BasicGemmWriterTemplate {
		if !self.eps.is_finite() || self.eps < 0.0
			|| !self.head_scale.value.is_finite() || !self.sep_scale.value.is_finite() {
			todo!("error for invalid values");
		}

		if self.head_dim % 32 != 0 || self.sep_dim % 32 != 0 {
			todo!("support RMSNorm GEMM epilogue dimensions that are not multiples of 32");
		}

		let chunk = self.head_dim + self.sep_dim;
		if !chunk.is_power_of_two() {
			todo!("assumed in `rrms_cols = b_rows >> chunk.trailing_zeros()`");
		}
		// Note: Other HEAD_DIM and SEP_DIM constraints depend on N_PER_WARP,
		// which is unknown at this point. We could get it from meta.json after the compilation,
		// but the compilation will fail if the constraints are not met.

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

	fn writer_template(&self, _c_cols: Option<NonZeroUsize>) -> BasicGemmWriterTemplate {
		todo!("support residual GEMM epilogues")
	}
}

impl EpilogueConfig for GeGluConfig {
	fn is_c_dtype_allowed(&self, c_dtype: DType) -> bool {
		c_dtype == DType::E4m3
	}

	fn writer_template(&self, c_cols: Option<NonZeroUsize>) -> BasicGemmWriterTemplate {
		if !self.inp_scale.value.is_finite() || self.inp_scale.value <= 0.0 {
			todo!("support invalid GeGLU input scale values");
		}
		if !self.out_scale.value.is_finite() || self.out_scale.value <= 0.0 {
			todo!("support invalid GeGLU output scale values");
		}
		let Some(c_cols) = c_cols else {
			todo!("support GeGLU GEMM outputs with runtime column count");
		};
		if c_cols.get() % 2 != 0 {
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
		kernel_build::compile_kernel_ptx(&kernel_path, &ptx_path, diag)?;
		kernel_build::compile_kernel_cubin(&ptx_path, &cubin_path, diag)?;
		kernel_build::compile_meta_exe(&meta_path, &meta_exe_path, diag)?;
		kernel_build::run_meta_exe(&meta_exe_path, &meta_json_path, diag)?;

		// TODO - if we have c_rows/c_cols, assert they are multiple of M_PER_BLOCK/N_PER_BLOCK

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
		kernel_build::write_generated_file(
			common_path,
			common.render().unwrap_or_else(|_| todo!("render GEMM common template")),
			diag,
		)?;

		let kernel = BasicGemmKernelTemplate {
			a_cols: gemm_config.a.cols,
			b_rows,
			writer: &writer_template,
		};
		kernel_build::write_generated_file(
			kernel_path,
			kernel.render().unwrap_or_else(|_| todo!("render GEMM kernel template")),
			diag,
		)?;

		let meta = BasicGemmMetaTemplate {};
		kernel_build::write_generated_file(
			meta_path,
			meta.render().unwrap_or_else(|_| todo!("render GEMM metadata template")),
			diag,
		)?;

		Ok(())
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
		metadata_path: &Path,
		diag: &mut Diagnostics,
	) -> Result<Self, KernelGeneratorError> {
		let metadata = kernel_build::read_metadata_json::<GemmLauncherMetadata>(
			metadata_path, diag
		)?;
		if metadata.THREADS_PER_BLOCK == 0
			|| !metadata.M_PER_BLOCK.is_power_of_two() // Note: this also ensures != 0
			|| !metadata.N_PER_BLOCK.is_power_of_two()
		{
			cold_path();
			let file = metadata_path.to_string_lossy();
			diag.add_error(format!("Invalid JSON metadata in {file}"));
			return Err(KernelGeneratorError);
		}

		let kernel = kernel_build::load_cubin_kernel(
			&device.stream, cubin_path, "kernel", metadata.SMEM_BYTES, diag,
		)?;

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

	fn launch_gemm(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let [a_rows, _a_cols] = tensor_check::matrix_shape(
			"GEMM", "a", a, [None, Some(self.a_cols)],
			&self.device, self.a_dtype
		)?;
		let [b_rows, _b_cols] = tensor_check::matrix_shape(
			"GEMM", "b", b, [self.b_rows.map(NonZeroUsize::get), Some(self.a_cols)],
			&self.device, self.b_dtype
		)?;
		let [c_rows, c_cols] = tensor_check::matrix_shape(
			"GEMM", "c", c, [Some(a_rows), Some(b_rows)],
			&self.device, self.c_dtype
		)?;

		debug_assert!(self.metadata.M_PER_BLOCK.is_power_of_two());
		debug_assert!(self.metadata.N_PER_BLOCK.is_power_of_two());
		if c_rows & (self.metadata.M_PER_BLOCK - 1) != 0
			|| c_cols & (self.metadata.N_PER_BLOCK - 1) != 0 {
			cold_path();
			return Err(ErrPack::new(TensorOpError::Other, format!(
				"GEMM output shape [{c_rows}, {c_cols}] must be divisible by tile shape [{}, {}]",
				self.metadata.M_PER_BLOCK, self.metadata.N_PER_BLOCK
			)));
		}

		let grid_x = c_rows >> self.metadata.M_PER_BLOCK.trailing_zeros();
		let grid_y = c_cols >> self.metadata.N_PER_BLOCK.trailing_zeros();
		let mut a_ptr = unsafe { a.device_ptr().as_ptr::<c_void>() };
		let mut a_rows_arg = a_rows;
		let mut b_ptr = unsafe { b.device_ptr().as_ptr::<c_void>() };
		let mut b_rows_arg = b_rows;
		let mut c_ptr = unsafe { c.device_ptr().as_ptr::<c_void>() };

		// TODO - use struct instead of array of pointers
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

		let [a_rows, _a_cols] = tensor_check::matrix_shape(
			"GEMM", "a", a, [None, Some(basic.a_cols)],
			&basic.device, basic.a_dtype
		)?;
		let [b_rows, _b_cols] = tensor_check::matrix_shape(
			"GEMM", "b", b, [basic.b_rows.map(NonZeroUsize::get), Some(basic.a_cols)],
			&basic.device, basic.b_dtype
		)?;
		if b_rows % 2 != 0 {
			cold_path();
			return Err(ErrPack::new(TensorOpError::Other, format!(
				"GeGLU GEMM requires an even raw output column count, got {b_rows}"
			)));
		}
		let [c_rows, c_cols] = tensor_check::matrix_shape(
			"GEMM", "c", c, [Some(a_rows), Some(b_rows / 2)],
			&basic.device, basic.c_dtype
		)?;

		debug_assert!(basic.metadata.M_PER_BLOCK.is_power_of_two());
		debug_assert!(basic.metadata.N_PER_BLOCK.is_power_of_two());
		if c_rows & (basic.metadata.M_PER_BLOCK - 1) != 0
			|| c_cols & (basic.metadata.N_PER_BLOCK - 1) != 0 {
			cold_path();
			return Err(ErrPack::new(TensorOpError::Other, format!(
				"GeGLU GEMM raw output shape [{c_rows}, {b_rows}] must be divisible by tile shape [{}, {}]",
				basic.metadata.M_PER_BLOCK, basic.metadata.N_PER_BLOCK
			)));
		}

		let grid_x = c_rows >> basic.metadata.M_PER_BLOCK.trailing_zeros();
		let grid_y = c_cols >> basic.metadata.N_PER_BLOCK.trailing_zeros();
		let mut a_ptr = unsafe { a.device_ptr().as_ptr::<c_void>() };
		let mut a_rows_arg = a_rows;
		let mut b_ptr = unsafe { b.device_ptr().as_ptr::<c_void>() };
		let mut b_rows_arg = b_rows;
		let mut c_ptr = unsafe { c.device_ptr().as_ptr::<c_void>() };

		// TODO - use struct instead of array of pointers
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

		let [a_rows, _a_cols] = tensor_check::matrix_shape(
			"GEMM", "a", a, [None, Some(basic.a_cols)],
			&basic.device, basic.a_dtype
		)?;
		let [b_rows, _b_cols] = tensor_check::matrix_shape(
			"GEMM", "b", b, [basic.b_rows.map(NonZeroUsize::get), Some(basic.a_cols)],
			&basic.device, basic.b_dtype
		)?;
		let [c_rows, c_cols] = tensor_check::matrix_shape(
			"GEMM", "c", c, [Some(a_rows), Some(b_rows)],
			&basic.device, basic.c_dtype
		)?;

		debug_assert!(basic.metadata.M_PER_BLOCK.is_power_of_two());
		debug_assert!(basic.metadata.N_PER_BLOCK.is_power_of_two());
		if c_rows & (basic.metadata.M_PER_BLOCK - 1) != 0
			|| c_cols & (basic.metadata.N_PER_BLOCK - 1) != 0 {
			cold_path();
			return Err(ErrPack::new(TensorOpError::Other, format!(
				"GEMM output shape [{c_rows}, {c_cols}] must be divisible by tile shape [{}, {}]",
				basic.metadata.M_PER_BLOCK, basic.metadata.N_PER_BLOCK
			)));
		}

		let chunk = head_dim + sep_dim;
		debug_assert!(chunk.is_power_of_two());
		let rrms_cols = b_rows >> chunk.trailing_zeros();
		tensor_check::matrix_shape(
			"GEMM", "rrms", rrms, [Some(a_rows), Some(rrms_cols)],
			&basic.device, DType::F32
		)?;

		let grid_x = c_rows >> basic.metadata.M_PER_BLOCK.trailing_zeros();
		let grid_y = c_cols >> basic.metadata.N_PER_BLOCK.trailing_zeros();
		let mut a_ptr = unsafe { a.device_ptr().as_ptr::<c_void>() };
		let mut a_rows_arg = a_rows;
		let mut b_ptr = unsafe { b.device_ptr().as_ptr::<c_void>() };
		let mut b_rows_arg = b_rows;
		let mut c_ptr = unsafe { c.device_ptr().as_ptr::<c_void>() };
		let mut rrms_ptr = unsafe { rrms.device_ptr().as_ptr::<c_void>() };

		// TODO - use struct instead of array of pointers
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

//--------------------------------------------------------------------------------------------------
