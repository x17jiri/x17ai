//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#![allow(warnings)] // TODO - disabling warnings for main.rs. Remove this later.
#![allow(non_snake_case)]
#![allow(clippy::manual_is_multiple_of)]
#![feature(generic_const_exprs)]
#![feature(macro_metavar_expr)]
#![feature(string_from_utf8_lossy_owned)]
#![feature(f16)]
#![feature(thin_box)]
#![feature(box_into_inner)]

use std::io::{Error, ErrorKind};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use x17ai::device::Device;
use x17ai::device::cuda::{
	CudaDevice, CudaTimer, Diagnostics, GemmEpilogue, GemmInput, GemmKernel, GemmKernelArgs,
	GemmKernelConfig, GemmKernelExtraArgs, RMSNormEpilogue, Scale,
};
use x17ai::dtype::DType;
use x17ai::tensor::Tensor;

const HEAD_DIM: usize = 32;
const N_HEADS: usize = 64;
const L2_NORM_EPS: f64 = 1.0 / (1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0);

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let device = CudaDevice::new(0)?;

	let x = Tensor::from_safetensors_file(tensor_path("x_i8.safetensors"), device.clone())?;
	let q_weights = Tensor::from_safetensors_file(
		tensor_path("attn_q_weights_i8.safetensors"),
		device.clone(),
	)?;
	let kv_weights = Tensor::from_safetensors_file(
		tensor_path("attn_kv_weights_i8.safetensors"),
		device.clone(),
	)?;
	let (n_inputs, model_dim, q_proj_outputs) = validate_attn_q_inputs(&x, &q_weights)?;
	let kv_proj_outputs = validate_attn_kv_inputs(&x, &kv_weights, model_dim)?;

	let q = Tensor::new_empty(&[n_inputs, q_proj_outputs], DType::Int8, device.clone())?;
	let kv = Tensor::new_empty(&[n_inputs, kv_proj_outputs], DType::Int8, device.clone())?;
	let k_rrms = Tensor::new_empty(&[n_inputs, N_HEADS], DType::F32, device.clone())?;
	let kernel_config = attn_q_kernel_config(model_dim, q_proj_outputs)?;
	let mut diagnostics = Diagnostics::new();
	let kernel = match GemmKernel::new(device.clone(), "attn_q_fwd", &kernel_config, &mut diagnostics) {
		Ok(kernel) => {
			print_diagnostics(&diagnostics);
			kernel
		},
		Err(_) => {
			print_diagnostics(&diagnostics);
			return Err(other_error(format!(
				"failed to generate attn_q_fwd kernel; {} error(s)",
				diagnostics.err_count
			)).into());
		},
	};

	let kv_kernel_config = attn_kv_kernel_config(model_dim, kv_proj_outputs, HEAD_DIM)?;
	let mut diagnostics = Diagnostics::new();
	let kv_kernel = match GemmKernel::new(device.clone(), "attn_kv_fwd", &kv_kernel_config, &mut diagnostics) {
		Ok(kernel) => {
			print_diagnostics(&diagnostics);
			kernel
		},
		Err(_) => {
			print_diagnostics(&diagnostics);
			return Err(other_error(format!(
				"failed to generate attn_kv_fwd kernel; {} error(s)",
				diagnostics.err_count
			)).into());
		},
	};

	let kernel_timer = CudaTimer::new(device.as_ref())?;
	kernel_timer.start()?;
	kernel.run(GemmKernelArgs {
		a: &x,
		b: &q_weights,
		c: &q,
		extra: GemmKernelExtraArgs::None,
	})?;
	kernel_timer.stop()?;
	let q = q.to_cpu()?;
	device.synchronize()?;
	let kernel_seconds = kernel_timer.elapsed_seconds()?;
	q.save_safetensors_file(output_path("q_i8.safetensors"))?;

	let kv_kernel_timer = CudaTimer::new(device.as_ref())?;
	kv_kernel_timer.start()?;
	kv_kernel.run(GemmKernelArgs {
		a: &x,
		b: &kv_weights,
		c: &kv,
		extra: GemmKernelExtraArgs::RMSNorm { rrms: &k_rrms },
	})?;
	kv_kernel_timer.stop()?;
	let kv = kv.to_cpu()?;
	let k_rrms = k_rrms.to_cpu()?;
	device.synchronize()?;
	let kv_kernel_seconds = kv_kernel_timer.elapsed_seconds()?;
	kv.save_safetensors_file(output_path("kv_i8.safetensors"))?;
	k_rrms.save_safetensors_file(output_path("k_rrms_f32.safetensors"))?;

	println!("loaded x: {:?} {:?}, {} bytes on CUDA", x.shape(), x.dtype(), x.bytes());
	println!(
		"loaded attn_q_weights: {:?} {:?}, {} bytes on CUDA",
		q_weights.shape(),
		q_weights.dtype(),
		q_weights.bytes()
	);
	println!(
		"loaded attn_kv_weights: {:?} {:?}, {} bytes on CUDA",
		kv_weights.shape(),
		kv_weights.dtype(),
		kv_weights.bytes()
	);
	println!("allocated q: {:?} {:?}, {} bytes copied to CPU", q.shape(), q.dtype(), q.bytes());
	println!("allocated kv: {:?} {:?}, {} bytes copied to CPU", kv.shape(), kv.dtype(), kv.bytes());
	println!("allocated k_rrms: {:?} {:?}, {} bytes copied to CPU", k_rrms.shape(), k_rrms.dtype(), k_rrms.bytes());
	println!(
		"attn_q dimensions: n_inputs={n_inputs}, model_dim={model_dim}, q_proj_outputs={q_proj_outputs}"
	);
	println!(
		"attn_kv dimensions: n_inputs={n_inputs}, model_dim={model_dim}, kv_proj_outputs={kv_proj_outputs}"
	);
	println!("generated GEMM kernel files in {}", kernel.dir_path.display());
	println!("common source: {}", kernel.common_path.display());
	println!("kernel source: {}", kernel.kernel_path.display());
	println!("kernel ptx: {}", kernel.ptx_path.display());
	println!("kernel cubin: {}", kernel.cubin_path.display());
	println!("metadata source: {}", kernel.meta_path.display());
	println!("metadata executable: {}", kernel.meta_exe_path.display());
	println!("metadata json: {}", kernel.meta_json_path.display());
	println!("generated KV GEMM kernel files in {}", kv_kernel.dir_path.display());
	println!("kv common source: {}", kv_kernel.common_path.display());
	println!("kv kernel source: {}", kv_kernel.kernel_path.display());
	println!("kv kernel ptx: {}", kv_kernel.ptx_path.display());
	println!("kv kernel cubin: {}", kv_kernel.cubin_path.display());
	println!("kv metadata source: {}", kv_kernel.meta_path.display());
	println!("kv metadata executable: {}", kv_kernel.meta_exe_path.display());
	println!("kv metadata json: {}", kv_kernel.meta_json_path.display());
	let kernel_ms = kernel_seconds * 1000.0;
	let kernel_tflops = kernel.n_ops(&x, &q_weights) / kernel_seconds / 1.0e12;
	println!("launched attn_q_fwd in {kernel_ms:.4} ms, {kernel_tflops:.3} TFLOPS");
	let kv_kernel_ms = kv_kernel_seconds * 1000.0;
	let kv_kernel_tflops = kv_kernel.n_ops(&x, &kv_weights) / kv_kernel_seconds / 1.0e12;
	println!("launched attn_kv_fwd in {kv_kernel_ms:.4} ms, {kv_kernel_tflops:.3} TFLOPS");
	println!("stored {}", output_path("q_i8.safetensors").display());
	println!("stored {}", output_path("kv_i8.safetensors").display());
	println!("stored {}", output_path("k_rrms_f32.safetensors").display());

	Ok(())
}

fn tensor_path(file_name: &str) -> PathBuf {
	Path::new("tmp").join("block_torch").join(file_name)
}

fn output_path(file_name: &str) -> PathBuf {
	Path::new("tmp").join("block_rust").join(file_name)
}

fn print_diagnostics(diagnostics: &Diagnostics) {
	for diagnostic in &diagnostics.list {
		if diagnostic.is_error {
			eprintln!("kernel error: {}", diagnostic.message);
		} else {
			println!("kernel diagnostic: {}", diagnostic.message);
		}
	}
}

fn validate_attn_q_inputs(
	x: &Tensor,
	q_weights: &Tensor,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
	if x.dtype() != DType::Int8 {
		return Err(invalid_data(format!("expected x dtype i8, got {}", x.dtype())).into());
	}
	if q_weights.dtype() != DType::Int8 {
		return Err(invalid_data(format!(
			"expected attn_q_weights dtype i8, got {}",
			q_weights.dtype()
		))
		.into());
	}
	if x.shape().len() != 2 {
		return Err(
			invalid_data(format!("expected x to be rank 2, got shape {:?}", x.shape())).into()
		);
	}
	if q_weights.shape().len() != 2 {
		return Err(invalid_data(format!(
			"expected attn_q_weights to be rank 2, got shape {:?}",
			q_weights.shape()
		))
		.into());
	}

	let n_inputs = x.shape()[0];
	let model_dim = x.shape()[1];
	let q_proj_outputs = q_weights.shape()[0];
	let weights_cols = q_weights.shape()[1];
	if n_inputs == 0 || model_dim == 0 || q_proj_outputs == 0 {
		return Err(invalid_data(format!(
			"expected non-empty attn_q tensors, got x={:?}, attn_q_weights={:?}",
			x.shape(),
			q_weights.shape()
		)).into());
	}
	if weights_cols != model_dim {
		return Err(invalid_data(format!(
			"expected attn_q_weights second dimension to match x model dim; got {weights_cols} vs {model_dim}"
		)).into());
	}
	if n_inputs % 64 != 0 {
		return Err(invalid_data(format!(
			"expected x rows to be a multiple of 64 for this kernel, got {n_inputs}"
		)).into());
	}
	if q_proj_outputs % 128 != 0 {
		return Err(invalid_data(format!(
			"expected attn_q output columns to be a multiple of 128 for this kernel, got {q_proj_outputs}"
		)).into());
	}

	Ok((n_inputs, model_dim, q_proj_outputs))
}

fn validate_attn_kv_inputs(
	x: &Tensor,
	kv_weights: &Tensor,
	model_dim: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
	if kv_weights.dtype() != DType::Int8 {
		return Err(invalid_data(format!(
			"expected attn_kv_weights dtype i8, got {}",
			kv_weights.dtype()
		)).into());
	}
	if kv_weights.shape().len() != 2 {
		return Err(invalid_data(format!(
			"expected attn_kv_weights to be rank 2, got shape {:?}",
			kv_weights.shape()
		)).into());
	}
	if x.shape().len() != 2 {
		return Err(
			invalid_data(format!("expected x to be rank 2, got shape {:?}", x.shape())).into()
		);
	}

	let kv_proj_outputs = kv_weights.shape()[0];
	let weights_cols = kv_weights.shape()[1];
	if weights_cols != model_dim {
		return Err(invalid_data(format!(
			"expected attn_kv_weights second dimension to match x model dim; got {weights_cols} vs {model_dim}"
		)).into());
	}

	let expected_outputs = 2 * N_HEADS * HEAD_DIM;
	if kv_proj_outputs != expected_outputs {
		return Err(invalid_data(format!(
			"expected attn_kv_weights first dimension to be {expected_outputs}, got {kv_proj_outputs}"
		)).into());
	}
	if kv_proj_outputs % 128 != 0 {
		return Err(invalid_data(format!(
			"expected attn_kv output columns to be a multiple of 128 for this kernel, got {kv_proj_outputs}"
		)).into());
	}
	Ok(kv_proj_outputs)
}

fn attn_q_kernel_config(
	model_dim: usize,
	q_proj_outputs: usize,
) -> Result<GemmKernelConfig, Box<dyn std::error::Error>> {
	let Some(q_proj_outputs) = NonZeroUsize::new(q_proj_outputs) else {
		return Err(invalid_data("expected non-zero q projection output count".to_owned()).into());
	};

	Ok(GemmKernelConfig {
		a: GemmInput {
			dtype: DType::Int8,
			cols: model_dim,
			rows: None,
			trans: false,
		},
		b: GemmInput {
			dtype: DType::Int8,
			cols: model_dim,
			rows: Some(q_proj_outputs),
			trans: true,
		},
		epilogue: GemmEpilogue::Scale(Scale {
			value: 1.0 / f64::sqrt(model_dim as f64),
			description: format!("1 / sqrt({model_dim})").into(),
		}),
		c_dtype: DType::Int8,
	})
}

fn attn_kv_kernel_config(
	model_dim: usize,
	kv_proj_outputs: usize,
	head_dim: usize,
) -> Result<GemmKernelConfig, Box<dyn std::error::Error>> {
	let Some(kv_proj_outputs) = NonZeroUsize::new(kv_proj_outputs) else {
		return Err(invalid_data("expected non-zero kv projection output count".to_owned()).into());
	};

	Ok(GemmKernelConfig {
		a: GemmInput {
			dtype: DType::Int8,
			cols: model_dim,
			rows: None,
			trans: false,
		},
		b: GemmInput {
			dtype: DType::Int8,
			cols: model_dim,
			rows: Some(kv_proj_outputs),
			trans: true,
		},
		epilogue: GemmEpilogue::RMSNorm(RMSNormEpilogue {
			eps: L2_NORM_EPS,
			head_dim,
			sep_dim: head_dim,
			head_scale: Scale {
				value: 1.0,
				description: "1".into(),
			},
			sep_scale: Scale {
				value: 1.0 / f64::sqrt(model_dim as f64),
				description: format!("1 / sqrt({model_dim})").into(),
			},
		}),
		c_dtype: DType::Int8,
	})
}

fn invalid_data(message: String) -> Error {
	Error::new(ErrorKind::InvalidData, message)
}

fn other_error(message: String) -> Error {
	Error::new(ErrorKind::Other, message)
}
