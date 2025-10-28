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
use crate::tensor::device::cuda::kernel_templates::{
	ElemwiseArgTemplate, ElemwiseTemplate, ReduceArgTemplate, ReduceTemplate,
};
use crate::tensor::device::kernel::{DynExpr, DynExprKind, DynKernelCall};
use crate::tensor::device::{
	AttentionArgs, DeviceBuffer, DevicePtr, MatMulArgs, NewDeviceBufferError,
};
use crate::tensor::{DType, Device, HasDType, TensorOpError, UnsupportedDTypeError};
use crate::util::hasher::HashWord;
use crate::util::{ToBoxedSlice, mycell};

pub mod cuda_shim;
pub mod kernel_templates;

//--------------------------------------------------------------------------------------------------

struct KernelCacheEntry {
	kernel1: Option<CudaKernel>,
	key: Box<[HashWord]>, // TODO - allocate `key` inline at the end of the struct
}

struct KernelCacheEntryRef {
	pub key_hash: u64,
	pub value: Box<KernelCacheEntry>,
}

pub struct CudaDevice {
	cuda_stream: CudaStream,
	hash_random_state: crate::util::hasher::RandomState,
	kernel_cache: RefCell<HashTable<KernelCacheEntryRef>>,
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
			kernel_cache: RefCell::new(HashTable::with_capacity(20)),
			name,
		}))
	}

	pub fn capability(&self) -> CudaCapability {
		self.cuda_stream.capability()
	}

	pub fn warp_size(&self) -> usize {
		self.cuda_stream.warp_size()
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

	fn print_expr(out: &mut String, expr: &DynExpr) -> std::fmt::Result {
		match &expr.kind {
			&DynExprKind::ElemwiseTensorArg(index) => {
				write!(out, "e{index}")
			},
			&DynExprKind::ReduceTensorArg(index) => {
				write!(out, "r{index}")
			},
			&DynExprKind::ScalarArg(index) => {
				write!(out, "s{index}")
			},

			DynExprKind::SumExpr(..) | DynExprKind::MaxExpr(..) => {
				write!(out, "reduce_val")
			},

			DynExprKind::NegExpr(inner) => {
				write!(out, "-")?;
				Self::print_expr(out, inner.as_ref())
			},
			DynExprKind::ExpExpr(inner) => {
				write!(out, "exp(")?;
				Self::print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},
			DynExprKind::LnExpr(inner) => {
				write!(out, "ln(")?;
				Self::print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},
			DynExprKind::AbsExpr(inner) => {
				write!(out, "abs(")?;
				Self::print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},
			DynExprKind::SqrtExpr(inner) => {
				write!(out, "sqrt(")?;
				Self::print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},
			DynExprKind::RecipExpr(inner) => {
				write!(out, "(1.0 / ")?;
				Self::print_expr(out, inner.as_ref())?;
				write!(out, ")")
			},

			DynExprKind::AddExpr(lhs, rhs) => {
				write!(out, "(")?;
				Self::print_expr(out, lhs.as_ref())?;
				write!(out, " + ")?;
				Self::print_expr(out, rhs.as_ref())?;
				write!(out, ")")
			},
			DynExprKind::SubExpr(lhs, rhs) => {
				write!(out, "(")?;
				Self::print_expr(out, lhs.as_ref())?;
				write!(out, " - ")?;
				Self::print_expr(out, rhs.as_ref())?;
				write!(out, ")")
			},
			DynExprKind::MulExpr(lhs, rhs) => {
				write!(out, "(")?;
				Self::print_expr(out, lhs.as_ref())?;
				write!(out, " * ")?;
				Self::print_expr(out, rhs.as_ref())?;
				write!(out, ")")
			},
		}
	}

	fn get_cache_entry<'cache>(
		&self,
		kernel_cache: &'cache mut HashTable<KernelCacheEntryRef>,
		key: &[HashWord],
	) -> &'cache mut KernelCacheEntry {
		let key_hash = self.hash_random_state.hash_one(key);
		let entry = kernel_cache.find_entry(key_hash, |item| {
			item.key_hash == key_hash && likely(item.value.key.as_ref() == key)
		});
		match entry {
			Ok(occupied_entry) => occupied_entry.into_mut().value.as_mut(),
			Err(absent_entry) => {
				cold_path();
				let new_cache_entry = KernelCacheEntryRef {
					key_hash,
					value: Box::new(KernelCacheEntry { kernel1: None, key: key.to_boxed_slice() }),
				};
				let rehash_hasher = |item: &KernelCacheEntryRef| item.key_hash;
				#[rustfmt::skip]
				absent_entry
					.into_table()
					.insert_unique(key_hash, new_cache_entry, rehash_hasher)
					.into_mut().value.as_mut()
			},
		}
	}

	/// This is similar to `Option::get_or_insert_with`,
	/// but works with functions that return `Result`.
	fn get_or(
		option: &mut Option<CudaKernel>,
		f: impl FnOnce() -> Result<CudaKernel, ErrPack<TensorOpError>>,
	) -> Result<&CudaKernel, ErrPack<TensorOpError>> {
		if option.is_none() {
			cold_path();
			*option = Some(f()?);
		}
		// SAFETY: we just ensured that the option is `Some`
		Ok(unsafe { option.as_ref().unwrap_unchecked() })
	}

	unsafe fn run_elemwise(
		&self,
		data: &DynKernelCall,
		cache_entry: &mut KernelCacheEntry,
	) -> Result<(), ErrPack<TensorOpError>> {
		let kernel = Self::get_or(&mut cache_entry.kernel1, || self.compile_elemwise(data))?;

		const BLOCK_SIZE: usize = 1024;
		let block_cnt = (data.output.size[0] * data.output.size[1]).div_ceil(BLOCK_SIZE);
		let config = CudaLaunchConfig {
			grid_dim: CudaCube { x: block_cnt, y: 1, z: 1 },
			block_dim: CudaCube { x: BLOCK_SIZE, y: 1, z: 1 },
			shared_mem_bytes: 0,
		};
		unsafe {
			let output_arg = std::ptr::from_ref(data.output).cast();
			let elemwise_args = std::ptr::from_ref(data.elemwise_args).cast();
			let scalar_args = std::ptr::from_ref(data.scalar_args).cast();
			self.cuda_stream.run_kernel(
				kernel,
				&config,
				// Note: Passing `elemwise_args` when it is empty could cause problems because C++
				// doesn't support zero-sized types.
				// So if it is empty, we pass `scalar_args` as the second argument. We also pass it
				// as the third argument, but the third position is unused in that case.
				// It is ok to pass more arguments than the kernel actually uses.
				&[
					output_arg,
					if data.elemwise_args.is_empty() { scalar_args } else { elemwise_args },
					scalar_args,
				],
			)?;
		}
		Ok(())
	}

	#[inline(never)]
	fn compile_elemwise(&self, data: &DynKernelCall) -> Result<CudaKernel, ErrPack<TensorOpError>> {
		let expr = data.generate_expr();
		let mut expr_str = String::new();
		Self::print_expr(&mut expr_str, &expr)
			.map_err(|e| CudaError::new(format!("Failed to render expression: {e}")))?;
		let mut src_data = ElemwiseTemplate {
			internal_dtype: data.internal_dtype().to_string(),
			out_dtype: data.output_dtype().to_string(),
			elem_args: Vec::with_capacity(data.elemwise_args.len()),
			scalar_args_count: data.scalar_args.len(),
			expr: expr_str,
		};
		for i in 0..data.elemwise_args.len() {
			let dtype = data.elemwise_dtype(i).to_string();
			src_data.elem_args.push(ElemwiseArgTemplate { dtype });
		}
		let kernel_src = src_data.to_string();
		println!("Rendered kernel source:\n{kernel_src}");
		let ptx = self.compile(&kernel_src)?;
		println!("Compiled PTX:\n{}", ptx.code());
		println!("Compilation log:\n{}", ptx.log());

		let module = self.cuda_stream.load_module(&ptx)?;
		let kernel = module.get_kernel("x17ai_kernel")?;
		Ok(kernel)
	}

	unsafe fn run_reduce(
		&self,
		data: &DynKernelCall,
		cache_entry: &mut KernelCacheEntry,
	) -> Result<(), ErrPack<TensorOpError>> {
		let kernel = Self::get_or(&mut cache_entry.kernel1, || self.compile_reduce(data))?;

		let WARP_SIZE = self.warp_size();
		let BLOCK_SIZE = 1024.min((data.output.reduction_size + WARP_SIZE - 1) & !(WARP_SIZE - 1));
		let block_cnt = data.output.size[0] * data.output.size[1];
		let config = CudaLaunchConfig {
			grid_dim: CudaCube { x: block_cnt, y: 1, z: 1 },
			block_dim: CudaCube { x: BLOCK_SIZE, y: 1, z: 1 },
			shared_mem_bytes: unsafe { data.internal_dtype().array_bytes_unchecked(WARP_SIZE) },
		};
		unsafe {
			let output_arg = std::ptr::from_ref(data.output).cast();
			let reduce_args = std::ptr::from_ref(data.reduce_args).cast();
			let elemwise_args = std::ptr::from_ref(data.elemwise_args).cast();
			let scalar_args = std::ptr::from_ref(data.scalar_args).cast();
			self.cuda_stream.run_kernel(
				kernel,
				&config,
				&[
					output_arg,
					reduce_args,
					if data.elemwise_args.is_empty() { scalar_args } else { elemwise_args },
					scalar_args,
				],
			)?;
		}
		Ok(())
	}

	#[inline(never)]
	fn compile_reduce(&self, data: &DynKernelCall) -> Result<CudaKernel, ErrPack<TensorOpError>> {
		let expr = data.generate_expr();
		let mut pre_reduce_expr = String::new();
		#[allow(clippy::unwrap_used)]
		Self::print_expr(&mut pre_reduce_expr, expr.pre_reduce().unwrap())
			.map_err(|e| CudaError::new(format!("Failed to render expression: {e}")))?;
		let mut post_reduce_expr = String::new();
		Self::print_expr(&mut post_reduce_expr, expr.as_ref())
			.map_err(|e| CudaError::new(format!("Failed to render expression: {e}")))?;
		let mut post_reduce_common = String::new();
		Self::print_expr(&mut post_reduce_common, expr.post_reduce_common())
			.map_err(|e| CudaError::new(format!("Failed to render expression: {e}")))?;
		let mut src_data = ReduceTemplate {
			internal_dtype: data.internal_dtype().to_string(),
			out_dtype: data.output_dtype().to_string(),
			reduce_args: Vec::with_capacity(data.reduce_args.len()),
			elem_args: Vec::with_capacity(data.elemwise_args.len()),
			scalar_args_count: data.scalar_args.len(),
			pre_reduce_expr,
			post_reduce_expr,
			post_reduce_common,
			zero: "0".to_string(), // TODO
			warp_size: self.warp_size(),
		};
		for i in 0..data.reduce_args.len() {
			let dtype = data.reduce_dtype(i).to_string();
			src_data.reduce_args.push(ReduceArgTemplate { dtype });
		}
		for i in 0..data.elemwise_args.len() {
			let dtype = data.elemwise_dtype(i).to_string();
			src_data.elem_args.push(ElemwiseArgTemplate { dtype });
		}
		let kernel_src = src_data.to_string();
		println!("Rendered kernel source:\n{kernel_src}");
		let ptx = self.compile(&kernel_src)?;
		println!("Compiled PTX:\n{}", ptx.code());
		println!("Compilation log:\n{}", ptx.log());

		let module = self.cuda_stream.load_module(&ptx)?;
		let kernel = module.get_kernel("x17ai_kernel")?;
		Ok(kernel)
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
		let Ok(mut kernel_cache) = self.kernel_cache.try_borrow_mut() else {
			cold_path();
			let err = CudaError::new("Internal error: compiled_kernels is already borrowed");
			return Err(err.into());
		};
		let cache_entry = self.get_cache_entry(&mut kernel_cache, data.key);
		if data.reduce_args.is_empty() {
			unsafe { self.run_elemwise(data, cache_entry) }
		} else {
			unsafe { self.run_reduce(data, cache_entry) }
		}
	}
}
