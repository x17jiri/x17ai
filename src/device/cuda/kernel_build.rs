//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::path::Path;
use std::process::{Command, Output};

use serde::de::DeserializeOwned;

use crate::{Diagnostics, KernelGeneratorError};

use super::cuda_shim::{CudaKernel, CudaStream};

//--------------------------------------------------------------------------------------------------

pub(super) fn write_generated_file(
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

pub(super) fn compile_kernel_ptx(
	kernel_path: &Path,
	ptx_path: &Path,
	diag: &mut Diagnostics,
) -> Result<(), KernelGeneratorError> {
	let mut command = nvcc_command("compute_86");
	command
		.arg("-ptx")
		.arg(kernel_path)
		.arg("-lineinfo")
		.arg("-o")
		.arg(ptx_path);

	run_checked_command(
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

pub(super) fn compile_kernel_cubin(
	ptx_path: &Path,
	cubin_path: &Path,
	diag: &mut Diagnostics,
) -> Result<(), KernelGeneratorError> {
	let mut command = nvcc_command("sm_86");
	command
		.arg("-Xptxas=-v")
		.arg("--cubin")
		.arg(ptx_path)
		.arg("-o")
		.arg(cubin_path);

	run_checked_command(
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

pub(super) fn compile_meta_exe(
	meta_path: &Path,
	meta_exe_path: &Path,
	diag: &mut Diagnostics,
) -> Result<(), KernelGeneratorError> {
	let mut command = nvcc_command("sm_86");
	command
		.arg(meta_path)
		.arg("-o")
		.arg(meta_exe_path);

	run_checked_command(
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

pub(super) fn run_meta_exe(
	meta_exe_path: &Path,
	meta_json_path: &Path,
	diag: &mut Diagnostics,
) -> Result<(), KernelGeneratorError> {
	let mut command = Command::new(meta_exe_path);
	let output = run_checked_command(
		&mut command,
		&format!("failed to run metadata executable {}", meta_exe_path.display()),
		diag,
	)?;

	match std::fs::write(meta_json_path, output.stdout) {
		Ok(()) => Ok(()),
		Err(err) => {
			cold_path();
			diag.add_error(format!(
				"failed to write generated metadata {}: {err}",
				meta_json_path.display(),
			));
			Err(KernelGeneratorError)
		},
	}
}

pub(super) fn read_metadata_json<T: DeserializeOwned>(
	metadata_path: &Path,
	diag: &mut Diagnostics,
) -> Result<T, KernelGeneratorError> {
	let metadata_json = match std::fs::read_to_string(metadata_path) {
		Ok(s) => s,
		Err(err) => {
			cold_path();
			let file = metadata_path.to_string_lossy();
			diag.add_error(format!("Error reading file {file}: {err}"));
			return Err(KernelGeneratorError);
		},
	};

	match serde_json::from_str::<T>(&metadata_json) {
		Ok(m) => Ok(m),
		Err(err) => {
			cold_path();
			let file = metadata_path.to_string_lossy();
			diag.add_error(format!("Failed to parse JSON data from {file}: {err}"));
			Err(KernelGeneratorError)
		},
	}
}

pub(super) fn load_cubin_kernel(
	stream: &CudaStream,
	cubin_path: &Path,
	kernel_name: &str,
	smem_size: usize,
	diag: &mut Diagnostics,
) -> Result<CudaKernel, KernelGeneratorError> {
	let module = match stream.load_module_from_cubin(cubin_path) {
		Ok(module) => module,
		Err(err) => {
			cold_path();
			let cubin_path = cubin_path.display();
			diag.add_error(format!("failed to load generated CUDA module `{cubin_path}`: {err}"));
			return Err(KernelGeneratorError);
		},
	};
	let kernel = match module.get_kernel(kernel_name, smem_size) {
		Ok(kernel) => kernel,
		Err(err) => {
			cold_path();
			let cubin_path = cubin_path.display();
			diag.add_error(format!(
				"failed to load generated CUDA kernel `{kernel_name:?}` from `{cubin_path}`: {err}"
			));
			return Err(KernelGeneratorError);
		},
	};
	Ok(kernel)
}

pub(super) fn nvcc_command(arch: &str) -> Command {
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

pub(super) fn run_checked_command(
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
