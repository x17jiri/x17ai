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
use crate::tensor::device::cuda::cuda_shim::{
	CudaCapability, CudaCppError, CudaCube, CudaError, CudaKernel, CudaLaunchConfig, CudaStream,
	Ptx, cuda_compile,
};
use crate::tensor::device::cuda::kernel_templates::{
	ElemwiseTemplate, ReduceTemplate, TensorArgTemplate,
};
use crate::tensor::device::kernel::{
	DynExpr, DynExprArgKind, DynExprBinaryKind, DynExprKind, DynExprReductionKind,
	DynExprUnaryKind, DynKernelCall,
};
use crate::tensor::device::{
	AttentionArgs, DevBufAllocFailedError, DeviceBuffer, DevicePtr, MatMulArgs,
};
use crate::tensor::{Device, TensorOpError};
use crate::util::ToBoxedSlice;
use crate::util::hasher::HashWord;
use crate::util::intrusive_rc::IntrusiveRc;

pub mod cuda_shim;
pub mod kernel_templates;

#[cfg(test)]
mod tests;

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

	fn print_expr(
		out: &mut String,
		expr: &DynExpr,
		replace: &[(&DynExpr, &str)],
	) -> std::fmt::Result {
		for (e, replacement) in replace.iter().copied() {
			if std::ptr::eq(e, expr) {
				return write!(out, "{replacement}");
			}
		}

		match &expr.kind {
			DynExprKind::Arg(a) => {
				match a.kind {
					DynExprArgKind::ElemwiseTensor => write!(out, "e"),
					DynExprArgKind::ReduceTensor => write!(out, "r"),
					DynExprArgKind::Scalar => write!(out, "s"),
				}?;
				write!(out, "{}", a.index)
			},
			DynExprKind::Reduction(..) => {
				write!(out, "reduce_val")
			},
			DynExprKind::Unary(un) => match un.kind {
				DynExprUnaryKind::Neg => {
					write!(out, "-")?;
					Self::print_expr(out, un.expr.as_ref(), replace)
				},
				DynExprUnaryKind::Exp => {
					write!(out, "exp(")?;
					Self::print_expr(out, un.expr.as_ref(), replace)?;
					write!(out, ")")
				},
				DynExprUnaryKind::Ln => {
					write!(out, "ln(")?;
					Self::print_expr(out, un.expr.as_ref(), replace)?;
					write!(out, ")")
				},
				DynExprUnaryKind::Abs => {
					write!(out, "abs(")?;
					Self::print_expr(out, un.expr.as_ref(), replace)?;
					write!(out, ")")
				},
				DynExprUnaryKind::Sqrt => {
					write!(out, "sqrt(")?;
					Self::print_expr(out, un.expr.as_ref(), replace)?;
					write!(out, ")")
				},
				DynExprUnaryKind::Recip => {
					write!(out, "(1.0 / ")?;
					Self::print_expr(out, un.expr.as_ref(), replace)?;
					write!(out, ")")
				},
			},
			DynExprKind::Binary(bin) => match bin.kind {
				DynExprBinaryKind::Add => {
					write!(out, "(")?;
					Self::print_expr(out, bin.lhs.as_ref(), replace)?;
					write!(out, " + ")?;
					Self::print_expr(out, bin.rhs.as_ref(), replace)?;
					write!(out, ")")
				},
				DynExprBinaryKind::Sub => {
					write!(out, "(")?;
					Self::print_expr(out, bin.lhs.as_ref(), replace)?;
					write!(out, " - ")?;
					Self::print_expr(out, bin.rhs.as_ref(), replace)?;
					write!(out, ")")
				},
				DynExprBinaryKind::Mul => {
					write!(out, "(")?;
					Self::print_expr(out, bin.lhs.as_ref(), replace)?;
					write!(out, " * ")?;
					Self::print_expr(out, bin.rhs.as_ref(), replace)?;
					write!(out, ")")
				},
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

		if data.output.size[0] != 1 {
			todo!("3d input not supported yet");
		}

		let total_elems = data.output.size[0] * data.output.size[1] * data.output.size[2];
		const BLOCK_SIZE: usize = 1024;
		let block_cnt = total_elems.div_ceil(BLOCK_SIZE);
		// TODO - `block_cnt` could exceed dim-x limit
		let config = CudaLaunchConfig {
			grid_dim: CudaCube { x: block_cnt, y: 1, z: 1 },
			block_dim: CudaCube { x: BLOCK_SIZE, y: 1, z: 1 },
			shared_mem_bytes: 0,
		};
		unsafe {
			let output_arg = std::ptr::from_ref(data.output).cast();
			let tensor_args = std::ptr::from_ref(data.tensor_args).cast();
			let scalar_args = std::ptr::from_ref(data.scalar_args).cast();
			self.cuda_stream.run_kernel(
				kernel,
				&config,
				// Note: Passing `tensor_args` when it is empty could cause problems because C++
				// doesn't support zero-sized types.
				// So if it is empty, we pass `scalar_args` as the second argument. We also pass it
				// as the third argument, but the third position is unused in that case.
				// It is ok to pass more arguments than the kernel actually uses.
				&[
					output_arg,
					if data.tensor_args.is_empty() { scalar_args } else { tensor_args },
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
		Self::print_expr(&mut expr_str, &expr, &[])
			.map_err(|e| CudaError::new(format!("Failed to render expression: {e}")))?;
		let mut src_data = ElemwiseTemplate {
			internal_dtype: data.internal_dtype().to_string(),
			out_dtype: data.output_dtype().to_string(),
			tensor_args: Vec::with_capacity(data.tensor_args.len()),
			scalar_args_count: data.scalar_args.len(),
			expr: expr_str,
		};
		for i in 0..data.tensor_args.len() {
			let dtype = data.arg_dtype(i).to_string();
			src_data.tensor_args.push(TensorArgTemplate { dtype });
		}
		let kernel_src = src_data.to_string();
		// println!("Rendered kernel source:\n{kernel_src}");
		let ptx = self.compile(&kernel_src)?;
		// println!("Compiled PTX:\n{}", ptx.code());
		// println!("Compilation log:\n{}", ptx.log());

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

		if data.output.size[0] != 1 {
			todo!("3d input not supported yet");
		}

		let cols = data.output.size[2];
		let rows = data.output.size[0] * data.output.size[1];
		let WARP_SIZE = self.warp_size();
		let BLOCK_SIZE = 1024.min((cols + WARP_SIZE - 1) & !(WARP_SIZE - 1));
		let block_cnt = rows; // TODO - `block_cnt` could exceed dim-x limit
		debug_assert!(BLOCK_SIZE % WARP_SIZE == 0);
		let config = CudaLaunchConfig {
			grid_dim: CudaCube { x: block_cnt, y: 1, z: 1 },
			block_dim: CudaCube { x: BLOCK_SIZE, y: 1, z: 1 },
			shared_mem_bytes: data.internal_dtype().ceil_bytes() * WARP_SIZE,
		};
		unsafe {
			let output_arg = std::ptr::from_ref(data.output).cast();
			let tensor_args = std::ptr::from_ref(data.tensor_args).cast();
			let scalar_args = std::ptr::from_ref(data.scalar_args).cast();
			self.cuda_stream.run_kernel(
				kernel,
				&config,
				&[
					output_arg,
					if data.tensor_args.is_empty() { scalar_args } else { tensor_args },
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
		let reduction = expr.find_reduction().unwrap();
		#[allow(clippy::unwrap_used)]
		Self::print_expr(&mut pre_reduce_expr, reduction.expr.as_ref(), &[])
			.map_err(|e| CudaError::new(format!("Failed to render expression: {e}")))?;
		let common_expr = expr.post_reduce_common();
		let mut post_reduce_expr = String::new();
		Self::print_expr(
			&mut post_reduce_expr,
			expr.as_ref(),
			&[(common_expr, "post_reduce_common")],
		)
		.map_err(|e| CudaError::new(format!("Failed to render expression: {e}")))?;
		let mut post_reduce_common = String::new();
		Self::print_expr(&mut post_reduce_common, common_expr, &[])
			.map_err(|e| CudaError::new(format!("Failed to render expression: {e}")))?;

		let (identity, loop_reduce, pairwise_reduce) = match reduction.kind {
			DynExprReductionKind::Sum => ("0", "KahanSum", "pairwise_sum"),
			DynExprReductionKind::Max => ("-inf()", "Max", "pairwise_max"),
		};

		let mut src_data = ReduceTemplate {
			internal_dtype: data.internal_dtype().to_string(),
			out_dtype: data.output_dtype().to_string(),
			tensor_args: Vec::with_capacity(data.tensor_args.len()),
			reduce_args_count: data.reduce_count,
			scalar_args_count: data.scalar_args.len(),
			pre_reduce_expr,
			post_reduce_expr,
			post_reduce_common,
			warp_size: self.warp_size(),
			identity,
			loop_reduce,
			pairwise_reduce,
		};
		for i in 0..data.tensor_args.len() {
			let dtype = data.arg_dtype(i).to_string();
			src_data.tensor_args.push(TensorArgTemplate { dtype });
		}
		let kernel_src = src_data.to_string();
		// println!("Rendered kernel source:\n{kernel_src}");
		let ptx = self.compile(&kernel_src)?;
		// println!("Compiled PTX:\n{}", ptx.code());
		// println!("Compilation log:\n{}", ptx.log());

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
		bytes: usize,
	) -> Result<IntrusiveRc<DeviceBuffer>, DevBufAllocFailedError> {
		let struct_layout = std::alloc::Layout::new::<DeviceBuffer>();
		unsafe {
			if let Some(inst_mem) = NonNull::new(std::alloc::alloc(struct_layout)) {
				if let Ok(memory) = self.cuda_stream.alloc(bytes) {
					let inst = inst_mem.cast::<DeviceBuffer>();
					inst.write(DeviceBuffer::new(memory, bytes, self));
					return Ok(IntrusiveRc::new(inst));
				}
				std::alloc::dealloc(inst_mem.as_ptr().cast(), struct_layout);
			}
			cold_path();
			Err(DevBufAllocFailedError)
		}
	}

	unsafe fn drop_buffer(
		self: Rc<Self>,
		device_ptr: DevicePtr,
		_bytes: usize,
		inst: NonNull<DeviceBuffer>,
	) {
		unsafe {
			self.cuda_stream.free(device_ptr);
			let struct_layout = std::alloc::Layout::new::<DeviceBuffer>();
			std::alloc::dealloc(inst.as_ptr().cast(), struct_layout);
		}
	}

	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: &DeviceBuffer,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { self.cuda_stream.upload_data(src, dst.device_ptr(), offset_bytes, count_bytes)? };
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
			self.cuda_stream.download_data(src.device_ptr(), dst, offset_bytes, count_bytes)?
		};
		Ok(())
	}

	unsafe fn mm(&self, _args: &MatMulArgs, _scale: f64) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement mm for CudaDevice");
	}

	unsafe fn attention(&self, _args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement attention for CudaDevice");
	}

	unsafe fn run_elemwise_kernel(
		&self,
		data: &DynKernelCall,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Ok(mut kernel_cache) = self.kernel_cache.try_borrow_mut() else {
			cold_path();
			let err = CudaError::new("Internal error: compiled_kernels is already borrowed");
			return Err(err.into());
		};
		let cache_entry = self.get_cache_entry(&mut kernel_cache, data.key);
		unsafe { self.run_elemwise(data, cache_entry) }
	}

	unsafe fn run_reduce_kernel(&self, data: &DynKernelCall) -> Result<(), ErrPack<TensorOpError>> {
		let Ok(mut kernel_cache) = self.kernel_cache.try_borrow_mut() else {
			cold_path();
			let err = CudaError::new("Internal error: compiled_kernels is already borrowed");
			return Err(err.into());
		};
		let cache_entry = self.get_cache_entry(&mut kernel_cache, data.key);
		unsafe { self.run_reduce(data, cache_entry) }
	}
}
