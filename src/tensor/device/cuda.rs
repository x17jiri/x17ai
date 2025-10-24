//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::fmt::Write;
use std::hint::{cold_path, likely};
use std::ptr::NonNull;
use std::rc::Rc;

use hashbrown::HashTable;

use crate::ErrPack;
use crate::tensor::device::cpu::cpu_float_methods::FromToF64;
use crate::tensor::device::cuda::cuda_shim::{
	CudaCapability, CudaCppError, CudaCube, CudaError, CudaKernel, CudaLaunchConfig, CudaStream,
	Ptx, cuda_compile,
};
use crate::tensor::device::cuda::kernel_templates::{Elemwise1DTemplate, ElemwiseArgTemplate};
use crate::tensor::device::kernel::{DynExpr, DynKernelCall};
use crate::tensor::device::{
	AttentionArgs, DeviceBuffer, DevicePtr, MatMulArgs, NewDeviceBufferError,
};
use crate::tensor::{DType, Device, HasDType, TensorOpError, UnsupportedDTypeError};
use crate::util::hasher::HashWord;
use crate::util::{ToBoxedSlice, mycell};

pub mod cuda_shim;
pub mod kernel_templates;

//--------------------------------------------------------------------------------------------------

struct CompiledKernel {
	d1: Option<CudaKernel>,
	d2: Option<CudaKernel>,
	key: Box<[HashWord]>, // TODO - allocate `key` inline at the end of the struct
}

struct CompiledKernelEntry {
	pub key_hash: u64,
	pub value: Box<CompiledKernel>,
}

pub struct CudaDevice {
	cuda_stream: CudaStream,
	hash_random_state: crate::util::hasher::RandomState,
	compiled_kernels: RefCell<HashTable<CompiledKernelEntry>>,
	name: String,
}

impl CudaDevice {
	pub fn new(device_id: usize) -> Result<Rc<Self>, CudaError> {
		Self::new_named(device_id, format!("CUDA Device {device_id}"))
	}

	pub fn new_named(device_id: usize, name: String) -> Result<Rc<Self>, CudaError> {
		Ok(Rc::new(Self {
			cuda_stream: CudaStream::new(device_id)?,
			hash_random_state: crate::util::hasher::RandomState::new(),
			compiled_kernels: RefCell::new(HashTable::with_capacity(20)),
			name,
		}))
	}

	pub fn capability(&self) -> CudaCapability {
		self.cuda_stream.capability()
	}

	pub fn compile(&self, src: &str) -> Result<Ptx, CudaCppError> {
		cuda_compile(self.capability(), src)
	}

	unsafe fn __read_float<T: HasDType + Default + FromToF64>(
		&self,
		buf: &DeviceBuffer,
		offset: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		debug_assert!(buf.dtype() == T::dtype);
		let val = T::default();
		unsafe {
			self.cuda_stream.download_data(
				buf.device_ptr(),
				NonNull::from(&val).cast(),
				offset,
				std::mem::size_of::<T>(),
			)?;
		}
		Ok(val.to_f64())
	}

	fn print_expr(&self, out: &mut String, expr: &DynExpr) -> std::fmt::Result {
		match expr {
			DynExpr::ElemwiseTensorArg(index) => {
				write!(out, "e{index}")
			},
			DynExpr::ReduceTensorArg(index) => {
				write!(out, "r{index}")
			},
			DynExpr::ScalarArg(index) => {
				write!(out, "s{index}")
			},

			DynExpr::NegExpr(inner) => {
				write!(out, "-")?;
				self.print_expr(out, inner.as_ref())
			},
			DynExpr::ExpExpr(inner) => {
				write!(out, "exp(")?;
				self.print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},
			DynExpr::LnExpr(inner) => {
				write!(out, "ln(")?;
				self.print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},
			DynExpr::AbsExpr(inner) => {
				write!(out, "abs(")?;
				self.print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},
			DynExpr::SqrtExpr(inner) => {
				write!(out, "sqrt(")?;
				self.print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},
			DynExpr::RecipExpr(inner) => {
				write!(out, "(1.0 / ")?;
				self.print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},

			DynExpr::AddExpr(lhs, rhs) => {
				write!(out, "(")?;
				self.print_expr(out, lhs.as_ref())?;
				write!(out, " + ")?;
				self.print_expr(out, rhs.as_ref())?;
				write!(out, ")")
			},
			DynExpr::SubExpr(lhs, rhs) => {
				write!(out, "(")?;
				self.print_expr(out, lhs.as_ref())?;
				write!(out, " - ")?;
				self.print_expr(out, rhs.as_ref())?;
				write!(out, ")")
			},
			DynExpr::MulExpr(lhs, rhs) => {
				write!(out, "(")?;
				self.print_expr(out, lhs.as_ref())?;
				write!(out, " * ")?;
				self.print_expr(out, rhs.as_ref())?;
				write!(out, ")")
			},
			_ => {
				cold_path();
				write!(out, "TODO")
			},
		}
	}

	fn compile_d1(&self, data: &DynKernelCall) -> Result<CudaKernel, ErrPack<TensorOpError>> {
		let expr = data.generate_expr();
		let mut expr_str = String::new();
		self.print_expr(&mut expr_str, &expr)
			.map_err(|e| CudaError::new(format!("Failed to render expression: {e}")))?;
		let mut src_data = Elemwise1DTemplate {
			internal_dtype: data.internal_dtype().to_string(),
			out_dtype: data.output_dtype().to_string(),
			elem_args: Vec::with_capacity(data.elemwise_args.len()),
			scalar_args_count: data.scalar_args.len(),
			expr: expr_str,
		};
		for i in 0..data.elemwise_args.len() {
			src_data.elem_args.push(ElemwiseArgTemplate {
				dtype: data.elemwise_dtype(i).to_string(),
			});
		}
		let kernel_src = src_data.to_string();
		println!("Rendered kernel source:\n{kernel_src}");
		let ptx = self.compile(&kernel_src)?;
		println!("Compiled PTX:\n{}", ptx.code());
		println!("Compilation log:\n{}", ptx.log());
		/*
		let module = self.cuda_stream.load_module(&ptx).ok().unwrap();
		let kernel = module.get_kernel("x17ai_kernel").ok().unwrap();
		*/
		todo!("... unfinished");
	}

	fn compile_d2(&self, _data: &DynKernelCall) -> Result<CudaKernel, ErrPack<TensorOpError>> {
		todo!("implement compile_d2 for CudaDevice");
	}

	fn get_kernel(&self, data: &DynKernelCall) -> Result<&CudaKernel, ErrPack<TensorOpError>> {
		let Ok(mut compiled_kernels) = self.compiled_kernels.try_borrow_mut() else {
			cold_path();
			let err = CudaError::new("Internal error: compiled_kernels is already borrowed");
			return Err(err.into());
		};
		let key = data.key;
		let key_hash = self.hash_random_state.hash_one(key);
		let found = compiled_kernels.find_mut(key_hash, |item| {
			item.key_hash == key_hash && likely(item.value.key.as_ref() == key)
		});
		let mut entry = //.
			if let Some(found) = found {
				NonNull::from_mut(found.value.as_mut())
			} else {
				cold_path();
				NonNull::from_mut(
					compiled_kernels
						.insert_unique(
							key_hash,
							CompiledKernelEntry {
								key_hash,
								value: Box::new(CompiledKernel {
									d1: None,
									d2: None,
									key: key.to_boxed_slice(),
								}),
							},
							|item| item.key_hash,
						)
						.get_mut().value.as_mut(),
				)
			};
		let entry = unsafe { entry.as_mut() };
		if data.output.size[0] == 1 {
			if entry.d1.is_none() {
				entry.d1 = Some(self.compile_d1(data)?);
			}
			Ok(unsafe { entry.d1.as_ref().unwrap_unchecked() })
		} else {
			if entry.d2.is_none() {
				entry.d2 = Some(self.compile_d2(data)?);
			}
			Ok(unsafe { entry.d2.as_ref().unwrap_unchecked() })
		}
	}
}

impl Device for CudaDevice {
	fn name(&self) -> &str {
		&self.name
	}

	#[inline(never)]
	fn new_buffer(
		self: Rc<Self>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<mycell::RefCell<DeviceBuffer>>, NewDeviceBufferError> {
		if let Some(size) = dtype.array_bytes(elems)
			&& let Ok(memory) = unsafe { self.cuda_stream.alloc(size) }
		{
			Ok(Rc::new(mycell::RefCell::new(unsafe {
				DeviceBuffer::new(memory, dtype, elems, self)
			})))
		} else {
			cold_path();
			Err(NewDeviceBufferError::AllocationFailed)
		}
	}

	unsafe fn drop_buffer(&self, device_ptr: DevicePtr, _dtype: DType, _elems: usize) {
		unsafe { self.cuda_stream.free(device_ptr) }
	}

	unsafe fn read_float(
		&self,
		buf: &DeviceBuffer,
		offset: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		match buf.dtype() {
			f32::dtype => unsafe { self.__read_float::<f32>(buf, offset) },
			f64::dtype => unsafe { self.__read_float::<f64>(buf, offset) },
			_ => {
				cold_path();
				Err(UnsupportedDTypeError.into())
			},
		}
	}

	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: &DeviceBuffer,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			self.cuda_stream.upload_data(src, dst.device_ptr(), offset_bytes, count_bytes)?;
		}
		Ok(())
	}

	unsafe fn download_data(
		&self,
		src: &DeviceBuffer,
		dst: NonNull<u8>,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			self.cuda_stream.download_data(src.device_ptr(), dst, offset_bytes, count_bytes)?;
		}
		Ok(())
	}

	unsafe fn mm(&self, _args: &MatMulArgs, _scale: f64) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement mm for CudaDevice");
	}

	unsafe fn attention(&self, _args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement attention for CudaDevice");
	}

	unsafe fn run_kernel(&self, data: &DynKernelCall) -> Result<(), ErrPack<TensorOpError>> {
		let kernel = self.get_kernel(data)?;
		let blocks = data.output.size[1].div_ceil(256);
		let config = CudaLaunchConfig {
			grid_dim: CudaCube { x: blocks, y: 1, z: 1 },
			block_dim: CudaCube { x: 256, y: 1, z: 1 },
			shared_mem_bytes: 0,
		};
		unsafe {
			self.cuda_stream.run_kernel(
				kernel,
				&config,
				&[
					std::ptr::from_ref(data.output).cast(),
					std::ptr::from_ref(data.elemwise_args).cast(),
				],
			)?;
		}
		Ok(())
	}
}
