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

use x17ai::device::cuda::{CudaDevice, CudaTimer, attn, gemm};
use x17ai::dtype::DType;
use x17ai::tensor::Tensor;
use x17ai::Diagnostics;

const HEAD_DIM: usize = 32;
const N_HEADS: usize = 64;
const F_WIDTH: usize = 2048;
const WINDOW_SIZE: usize = 0;
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
	let ffn_f_weights = Tensor::from_safetensors_file(
		tensor_path("ffn_f_weights_i8.safetensors"),
		device.clone(),
	)?;
	let sinks_k = Tensor::from_safetensors_file(
		tensor_path("sinks_k_i8.safetensors"),
		device.clone(),
	)?;
	let sinks_v = Tensor::from_safetensors_file(
		tensor_path("sinks_v_i8.safetensors"),
		device.clone(),
	)?;
	let attn_temperature = Tensor::from_safetensors_file(
		tensor_path("attn_temperature_f32.safetensors"),
		device.clone(),
	)?;
	let (n_inputs, model_dim, q_proj_outputs) = validate_attn_q_inputs(&x, &q_weights)?;
	let kv_proj_outputs = validate_attn_kv_inputs(&x, &kv_weights, model_dim)?;
	let ffn_f_proj_outputs = validate_ffn_f_inputs(&x, &ffn_f_weights, model_dim)?;
	let kv_weights = kv_weights.reshape(&[kv_proj_outputs, model_dim])?;

	let q = Tensor::new_empty(&[n_inputs, q_proj_outputs], DType::Int8, device.clone())?;
	let kv = Tensor::new_empty(&[n_inputs, kv_proj_outputs], DType::Int8, device.clone())?;
	let k_rrms = Tensor::new_empty(&[n_inputs, N_HEADS], DType::F32, device.clone())?;
	let attn_out = Tensor::new_empty(&[n_inputs, q_proj_outputs], DType::Int8, device.clone())?;
	let attn_l = Tensor::new_empty(&[N_HEADS, n_inputs], DType::F32, device.clone())?;
	let ffn_f = Tensor::new_empty(&[n_inputs, F_WIDTH], DType::E4m3, device.clone())?;

	let mut q_diagnostics = Diagnostics::new();
	let attn_q_kernel = match create_attn_q_kernel(
		device.clone(),
		model_dim,
		q_proj_outputs,
		&mut q_diagnostics,
	) {
		Ok(kernel) => {
			print_diagnostics(&q_diagnostics);
			kernel
		},
		Err(err) => {
			print_diagnostics(&q_diagnostics);
			return Err(err);
		},
	};

	let mut kv_diagnostics = Diagnostics::new();
	let attn_kv_kernel = match create_attn_kv_kernel(
		device.clone(),
		model_dim,
		kv_proj_outputs,
		HEAD_DIM,
		&mut kv_diagnostics,
	) {
		Ok(kernel) => {
			print_diagnostics(&kv_diagnostics);
			kernel
		},
		Err(err) => {
			print_diagnostics(&kv_diagnostics);
			return Err(err);
		},
	};

	let mut ffn_f_diagnostics = Diagnostics::new();
	let mut attn_diagnostics = Diagnostics::new();
	let attn_kernel = match create_attn_kernel(
		device.clone(),
		q_proj_outputs,
		kv_proj_outputs,
		q_proj_outputs,
		&mut attn_diagnostics,
	) {
		Ok(kernel) => {
			print_diagnostics(&attn_diagnostics);
			kernel
		},
		Err(err) => {
			print_diagnostics(&attn_diagnostics);
			return Err(err);
		},
	};

	let ffn_f_kernel = match create_ffn_f_kernel(
		device.clone(),
		model_dim,
		ffn_f_proj_outputs,
		&mut ffn_f_diagnostics,
	) {
		Ok(kernel) => {
			print_diagnostics(&ffn_f_diagnostics);
			kernel
		},
		Err(err) => {
			print_diagnostics(&ffn_f_diagnostics);
			return Err(err);
		},
	};

	let q_timer = CudaTimer::new(device.as_ref())?;
	q_timer.start()?;
	attn_q_kernel.launch(gemm::GemmArgs {
		a: &x,
		b: &q_weights,
		c: &q,
		extra: gemm::NoExtraArgs::new(),
	})?;
	q_timer.stop()?;

	let kv_timer = CudaTimer::new(device.as_ref())?;
	kv_timer.start()?;
	attn_kv_kernel.launch(gemm::GemmArgs {
		a: &x,
		b: &kv_weights,
		c: &kv,
		extra: gemm::RMSNormExtraArgs::new(&k_rrms),
	})?;
	kv_timer.stop()?;

	let attn_timer = CudaTimer::new(device.as_ref())?;
	attn_timer.start()?;
	attn_kernel.launch(attn::AttnKernelArgs {
		q: &q,
		kv: &kv,
		sink_k: &sinks_k,
		sink_v: &sinks_v,
		attn_temperature: &attn_temperature,
		out: &attn_out,
		l: Some(&attn_l),
		window_size: WINDOW_SIZE,
	})?;
	attn_timer.stop()?;

	let ffn_f_timer = CudaTimer::new(device.as_ref())?;
	ffn_f_timer.start()?;
	ffn_f_kernel.launch(gemm::GemmArgs {
		a: &x,
		b: &ffn_f_weights,
		c: &ffn_f,
		extra: gemm::NoExtraArgs::new(),
	})?;
	ffn_f_timer.stop()?;

	let q = q.to_cpu()?;
	let kv = kv.to_cpu()?.reshape(&[n_inputs, N_HEADS, 2, HEAD_DIM])?;
	let k_rrms = k_rrms.to_cpu()?;
	let attn_out = attn_out.to_cpu()?;
	let attn_l = attn_l.to_cpu()?;
	let ffn_f = ffn_f.to_cpu()?;
	device.synchronize()?;
	let q_kernel_seconds = q_timer.elapsed_seconds()?;
	let kv_kernel_seconds = kv_timer.elapsed_seconds()?;
	let attn_kernel_seconds = attn_timer.elapsed_seconds()?;
	let ffn_f_kernel_seconds = ffn_f_timer.elapsed_seconds()?;

	q.save_safetensors_file(output_path("q_i8.safetensors"))?;
	kv.save_safetensors_file(output_path("kv_i8.safetensors"))?;
	k_rrms.save_safetensors_file(output_path("k_rrms_f32.safetensors"))?;
	attn_out.save_safetensors_file(output_path("attn_out_i8.safetensors"))?;
	attn_l.save_safetensors_file(output_path("attn_l_f32.safetensors"))?;
	ffn_f.save_safetensors_file(output_path("ffn_f_f8.safetensors"))?;

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
	println!(
		"loaded ffn_f_weights: {:?} {:?}, {} bytes on CUDA",
		ffn_f_weights.shape(),
		ffn_f_weights.dtype(),
		ffn_f_weights.bytes()
	);
	println!(
		"loaded sinks_k: {:?} {:?}, {} bytes on CUDA",
		sinks_k.shape(),
		sinks_k.dtype(),
		sinks_k.bytes()
	);
	println!(
		"loaded sinks_v: {:?} {:?}, {} bytes on CUDA",
		sinks_v.shape(),
		sinks_v.dtype(),
		sinks_v.bytes()
	);
	println!(
		"loaded attn_temperature: {:?} {:?}, {} bytes on CUDA",
		attn_temperature.shape(),
		attn_temperature.dtype(),
		attn_temperature.bytes()
	);
	println!("allocated q: {:?} {:?}, {} bytes copied to CPU", q.shape(), q.dtype(), q.bytes());
	println!("allocated kv: {:?} {:?}, {} bytes copied to CPU", kv.shape(), kv.dtype(), kv.bytes());
	println!(
		"allocated k_rrms: {:?} {:?}, {} bytes copied to CPU",
		k_rrms.shape(),
		k_rrms.dtype(),
		k_rrms.bytes()
	);
	println!(
		"allocated attn_out: {:?} {:?}, {} bytes copied to CPU",
		attn_out.shape(),
		attn_out.dtype(),
		attn_out.bytes()
	);
	println!(
		"allocated attn_l: {:?} {:?}, {} bytes copied to CPU",
		attn_l.shape(),
		attn_l.dtype(),
		attn_l.bytes()
	);
	println!(
		"allocated ffn_f: {:?} {:?}, {} bytes copied to CPU",
		ffn_f.shape(),
		ffn_f.dtype(),
		ffn_f.bytes()
	);
	println!(
		"attn_q dimensions: n_inputs={n_inputs}, model_dim={model_dim}, q_proj_outputs={q_proj_outputs}"
	);
	println!(
		"attn_kv dimensions: n_inputs={n_inputs}, model_dim={model_dim}, kv_proj_outputs={kv_proj_outputs}"
	);
	println!(
		"ffn_f dimensions: n_inputs={n_inputs}, model_dim={model_dim}, ffn_f_proj_outputs={ffn_f_proj_outputs}"
	);
	println!("generated GEMM kernel files in {}", attn_q_kernel.dir_path.display());
	println!("common source: {}", attn_q_kernel.common_path.display());
	println!("kernel source: {}", attn_q_kernel.kernel_path.display());
	println!("kernel ptx: {}", attn_q_kernel.ptx_path.display());
	println!("kernel cubin: {}", attn_q_kernel.cubin_path.display());
	println!("metadata source: {}", attn_q_kernel.meta_path.display());
	println!("metadata executable: {}", attn_q_kernel.meta_exe_path.display());
	println!("metadata json: {}", attn_q_kernel.meta_json_path.display());
	println!("generated KV GEMM kernel files in {}", attn_kv_kernel.dir_path.display());
	println!("kv common source: {}", attn_kv_kernel.common_path.display());
	println!("kv kernel source: {}", attn_kv_kernel.kernel_path.display());
	println!("kv kernel ptx: {}", attn_kv_kernel.ptx_path.display());
	println!("kv kernel cubin: {}", attn_kv_kernel.cubin_path.display());
	println!("kv metadata source: {}", attn_kv_kernel.meta_path.display());
	println!("kv metadata executable: {}", attn_kv_kernel.meta_exe_path.display());
	println!("kv metadata json: {}", attn_kv_kernel.meta_json_path.display());
	println!("generated FFN F GEMM kernel files in {}", ffn_f_kernel.dir_path.display());
	println!("ffn_f common source: {}", ffn_f_kernel.common_path.display());
	println!("ffn_f kernel source: {}", ffn_f_kernel.kernel_path.display());
	println!("ffn_f kernel ptx: {}", ffn_f_kernel.ptx_path.display());
	println!("ffn_f kernel cubin: {}", ffn_f_kernel.cubin_path.display());
	println!("ffn_f metadata source: {}", ffn_f_kernel.meta_path.display());
	println!("ffn_f metadata executable: {}", ffn_f_kernel.meta_exe_path.display());
	println!("ffn_f metadata json: {}", ffn_f_kernel.meta_json_path.display());
	println!("generated attention kernel files in {}", attn_kernel.dir_path.display());
	println!("attention common source: {}", attn_kernel.common_path.display());
	println!("attention kernel source: {}", attn_kernel.kernel_path.display());
	println!("attention kernel ptx: {}", attn_kernel.ptx_path.display());
	println!("attention kernel cubin: {}", attn_kernel.cubin_path.display());
	println!("attention metadata source: {}", attn_kernel.meta_path.display());
	println!("attention metadata executable: {}", attn_kernel.meta_exe_path.display());
	println!("attention metadata json: {}", attn_kernel.meta_json_path.display());
	let q_kernel_ms = q_kernel_seconds * 1000.0;
	let q_kernel_tflops = attn_q_kernel.n_ops(&x, &q_weights) / q_kernel_seconds / 1.0e12;
	println!("launched attn_q_fwd in {q_kernel_ms:.4} ms, {q_kernel_tflops:.3} TFLOPS");
	let kv_kernel_ms = kv_kernel_seconds * 1000.0;
	let kv_kernel_tflops = attn_kv_kernel.n_ops(&x, &kv_weights) / kv_kernel_seconds / 1.0e12;
	println!("launched attn_kv_fwd in {kv_kernel_ms:.4} ms, {kv_kernel_tflops:.3} TFLOPS");
	let attn_kernel_ms = attn_kernel_seconds * 1000.0;
	let attn_kernel_tflops = attn_kernel.n_ops(n_inputs, WINDOW_SIZE) / attn_kernel_seconds / 1.0e12;
	println!("launched attn_fwd_i8 in {attn_kernel_ms:.4} ms, {attn_kernel_tflops:.3} TFLOPS");
	let ffn_f_kernel_ms = ffn_f_kernel_seconds * 1000.0;
	let ffn_f_kernel_tflops = ffn_f_kernel.n_ops(&x, &ffn_f_weights) / ffn_f_kernel_seconds / 1.0e12;
	println!("launched ffn_f_fwd in {ffn_f_kernel_ms:.4} ms, {ffn_f_kernel_tflops:.3} TFLOPS");
	println!("stored {}", output_path("q_i8.safetensors").display());
	println!("stored {}", output_path("kv_i8.safetensors").display());
	println!("stored {}", output_path("k_rrms_f32.safetensors").display());
	println!("stored {}", output_path("attn_out_i8.safetensors").display());
	println!("stored {}", output_path("attn_l_f32.safetensors").display());
	println!("stored {}", output_path("ffn_f_f8.safetensors").display());

	Ok(())
}

fn create_attn_q_kernel(
	device: Rc<CudaDevice>,
	model_dim: usize,
	q_proj_outputs: usize,
	diagnostics: &mut Diagnostics,
) -> Result<gemm::GemmKernel<gemm::ScaleEpilogue>, Box<dyn std::error::Error>> {
	let Some(q_proj_outputs) = NonZeroUsize::new(q_proj_outputs) else {
		return Err(invalid_data("expected non-zero q projection output count".to_owned()).into());
	};

	let gemm_config = gemm::GemmConfig {
		a: gemm::GemmInputConfig {
			dtype: DType::Int8,
			cols: model_dim,
			rows: None,
			trans: false,
		},
		b: gemm::GemmInputConfig {
			dtype: DType::Int8,
			cols: model_dim,
			rows: Some(q_proj_outputs),
			trans: true,
		},
		c_dtype: DType::Int8,
	};
	let scale_config = gemm::ScaleConfig(gemm::Scale {
		value: 1.0 / f64::sqrt(model_dim as f64),
		description: format!("1 / sqrt({model_dim})").into(),
	});

	match gemm::GemmKernel::<gemm::ScaleEpilogue>::new(
		device,
		"attn_q_fwd",
		&gemm_config,
		&scale_config,
		diagnostics,
	) {
		Ok(kernel) => Ok(kernel),
		Err(_) => Err(other_error(format!(
			"failed to generate attn_q_fwd kernel; {} error(s)",
			diagnostics.err_count
		)).into()),
	}
}

fn create_attn_kv_kernel(
	device: Rc<CudaDevice>,
	model_dim: usize,
	kv_proj_outputs: usize,
	head_dim: usize,
	diagnostics: &mut Diagnostics,
) -> Result<gemm::GemmKernel<gemm::RMSNormEpilogue>, Box<dyn std::error::Error>> {
	let Some(kv_proj_outputs) = NonZeroUsize::new(kv_proj_outputs) else {
		return Err(invalid_data("expected non-zero kv projection output count".to_owned()).into());
	};

	let gemm_config = gemm::GemmConfig {
		a: gemm::GemmInputConfig {
			dtype: DType::Int8,
			cols: model_dim,
			rows: None,
			trans: false,
		},
		b: gemm::GemmInputConfig {
			dtype: DType::Int8,
			cols: model_dim,
			rows: Some(kv_proj_outputs),
			trans: true,
		},
		c_dtype: DType::Int8,
	};
	let rms_norm_config = gemm::RMSNormConfig {
		eps: L2_NORM_EPS,
		head_dim,
		sep_dim: head_dim,
		head_scale: gemm::Scale {
			value: 1.0,
			description: "1".into(),
		},
		sep_scale: gemm::Scale {
			value: 1.0 / f64::sqrt(model_dim as f64),
			description: format!("1 / sqrt({model_dim})").into(),
		},
	};

	match gemm::GemmKernel::<gemm::RMSNormEpilogue>::new(
		device,
		"attn_kv_fwd",
		&gemm_config,
		&rms_norm_config,
		diagnostics,
	) {
		Ok(kernel) => Ok(kernel),
		Err(_) => Err(other_error(format!(
			"failed to generate attn_kv_fwd kernel; {} error(s)",
			diagnostics.err_count
		)).into()),
	}
}

fn create_attn_kernel(
	device: Rc<CudaDevice>,
	q_stride: usize,
	kv_stride: usize,
	o_stride: usize,
	diagnostics: &mut Diagnostics,
) -> Result<attn::AttnKernel, Box<dyn std::error::Error>> {
	let config = attn::AttnKernelConfig {
		dtype: DType::Int8,
		n_heads: N_HEADS,
		head_dim: HEAD_DIM,
		q_stride,
		kv_stride,
		o_stride,
	};

	match attn::AttnKernel::new(
		device,
		"attn_fwd_i8",
		config,
		diagnostics,
	) {
		Ok(kernel) => Ok(kernel),
		Err(_) => Err(other_error(format!(
			"failed to generate attn_fwd_i8 kernel; {} error(s)",
			diagnostics.err_count
		)).into()),
	}
}

fn create_ffn_f_kernel(
	device: Rc<CudaDevice>,
	model_dim: usize,
	ffn_f_proj_outputs: usize,
	diagnostics: &mut Diagnostics,
) -> Result<gemm::GemmKernel<gemm::GeGluEpilogue>, Box<dyn std::error::Error>> {
	let Some(ffn_f_proj_outputs) = NonZeroUsize::new(ffn_f_proj_outputs) else {
		return Err(invalid_data("expected non-zero FFN F projection output count".to_owned()).into());
	};

	let gemm_config = gemm::GemmConfig {
		a: gemm::GemmInputConfig {
			dtype: DType::Int8,
			cols: model_dim,
			rows: None,
			trans: false,
		},
		b: gemm::GemmInputConfig {
			dtype: DType::Int8,
			cols: model_dim,
			rows: Some(ffn_f_proj_outputs),
			trans: true,
		},
		c_dtype: DType::E4m3,
	};
	let geglu_config = gemm::GeGluConfig {
		inp_scale: gemm::Scale {
			value: 1.0 / f64::sqrt(model_dim as f64),
			description: format!("1 / sqrt({model_dim})").into(),
		},
		out_scale: gemm::Scale {
			value: 1.0 / f64::sqrt(model_dim as f64),
			description: format!("1 / sqrt({model_dim})").into(),
		},
	};

	match gemm::GemmKernel::<gemm::GeGluEpilogue>::new(
		device,
		"ffn_f_fwd",
		&gemm_config,
		&geglu_config,
		diagnostics,
	) {
		Ok(kernel) => Ok(kernel),
		Err(_) => Err(other_error(format!(
			"failed to generate ffn_f_fwd kernel; {} error(s)",
			diagnostics.err_count
		)).into()),
	}
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
	if x.shape().len() != 2 {
		return Err(
			invalid_data(format!("expected x to be rank 2, got shape {:?}", x.shape())).into()
		);
	}

	let shape = kv_weights.shape();
	let (kv_proj_outputs, weights_cols) = match shape {
		&[rows, cols] => (rows, cols),
		&[heads, parts, head_dim, cols] => {
			if heads != N_HEADS || parts != 2 || head_dim != HEAD_DIM {
				return Err(invalid_data(format!(
					"expected attn_kv_weights shape [{N_HEADS}, 2, {HEAD_DIM}, {model_dim}], got {shape:?}"
				)).into());
			}
			(heads * parts * head_dim, cols)
		},
		_ => {
			return Err(invalid_data(format!(
				"expected attn_kv_weights to be rank 2 or 4, got shape {shape:?}"
			)).into());
		},
	};
	if weights_cols != model_dim {
		return Err(invalid_data(format!(
			"expected attn_kv_weights final dimension to match x model dim; got {weights_cols} vs {model_dim}"
		)).into());
	}

	let expected_outputs = 2 * N_HEADS * HEAD_DIM;
	if kv_proj_outputs != expected_outputs {
		return Err(invalid_data(format!(
			"expected attn_kv_weights to contain {expected_outputs} output rows, got {kv_proj_outputs}"
		)).into());
	}
	if kv_proj_outputs % 128 != 0 {
		return Err(invalid_data(format!(
			"expected attn_kv output columns to be a multiple of 128 for this kernel, got {kv_proj_outputs}"
		)).into());
	}
	Ok(kv_proj_outputs)
}

fn validate_ffn_f_inputs(
	x: &Tensor,
	ffn_f_weights: &Tensor,
	model_dim: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
	if ffn_f_weights.dtype() != DType::Int8 {
		return Err(invalid_data(format!(
			"expected ffn_f_weights dtype i8, got {}",
			ffn_f_weights.dtype()
		)).into());
	}
	if x.shape().len() != 2 {
		return Err(
			invalid_data(format!("expected x to be rank 2, got shape {:?}", x.shape())).into()
		);
	}
	if ffn_f_weights.shape().len() != 2 {
		return Err(invalid_data(format!(
			"expected ffn_f_weights to be rank 2, got shape {:?}",
			ffn_f_weights.shape()
		)).into());
	}

	let ffn_f_proj_outputs = ffn_f_weights.shape()[0];
	let weights_cols = ffn_f_weights.shape()[1];
	if weights_cols != model_dim {
		return Err(invalid_data(format!(
			"expected ffn_f_weights second dimension to match x model dim; got {weights_cols} vs {model_dim}"
		)).into());
	}

	let expected_outputs = 2 * F_WIDTH;
	if ffn_f_proj_outputs != expected_outputs {
		return Err(invalid_data(format!(
			"expected ffn_f_weights first dimension to be {expected_outputs}, got {ffn_f_proj_outputs}"
		)).into());
	}
	if ffn_f_proj_outputs % 128 != 0 {
		return Err(invalid_data(format!(
			"expected ffn_f raw output columns to be a multiple of 128 for this kernel, got {ffn_f_proj_outputs}"
		)).into());
	}
	Ok(ffn_f_proj_outputs)
}

fn invalid_data(message: String) -> Error {
	Error::new(ErrorKind::InvalidData, message)
}

fn other_error(message: String) -> Error {
	Error::new(ErrorKind::Other, message)
}
