use askama::Template;

use crate::{device::cuda::{CudaDevice, GemmKernelLauncher}, tensor::Tensor, ErrPack, TensorOpError};

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

pub struct BasicGemmMetadata {
	pub threads_per_block: usize,
	pub smem_bytes: usize,
	pub m_per_block: usize,
	pub n_per_block: usize,
}

pub struct BasicGemmKernelLauncher;

impl GemmKernelLauncher for BasicGemmKernelLauncher {
	fn launch(
		&self, _device: &CudaDevice, _a: &Tensor, _b: &Tensor, _c: &Tensor
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!();
	}
}
