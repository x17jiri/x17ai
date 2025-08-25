//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::device::buffer::{
	DeviceBufferRef, DeviceBufferRefMut, DeviceBufferVMT, KernelElemArg, KernelOutput,
	KernelReduceArg,
};
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::executor::ExecutorError;
use crate::tensor::device::kernel::expr::DynExpr;
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::generic::map::{IndexToOffset, Map, ND, Select};
use crate::tensor::{HasDType, generic};

use super::math::{self, FromToF64};

//--------------------------------------------------------------------------------------------------

/*trait Slice2D<T> {
	fn slice(&self, dim0: usize, dim1: std::ops::RangeFull) -> &[T];
}

impl<T: Copy + HasDType + FromToF64> Slice2D<T> for generic::Tensor<ND<2>, &[T]> {
	fn slice(&self, dim0: usize, _dim1: std::ops::RangeFull) -> &[T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		self.buf().get(span).unwrap()
	}
}*/

trait Slice3D<T> {
	fn slice(&self, dim0: usize, dim1: usize, dim2: std::ops::RangeFull) -> &[T];
}

impl<T: Copy + HasDType + FromToF64> Slice3D<T> for generic::Tensor<ND<3>, &[T]> {
	fn slice(&self, dim0: usize, dim1: usize, _dim2: std::ops::RangeFull) -> &[T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let map = map.select(0, dim1).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		self.buf().get(span).unwrap()
	}
}

trait SliceMut2D<T> {
	fn slice_mut(&mut self, dim0: usize, dim1: std::ops::RangeFull) -> &mut [T];
}

impl<T: Copy + HasDType + FromToF64> SliceMut2D<T> for generic::Tensor<ND<2>, &mut [T]> {
	fn slice_mut(&mut self, dim0: usize, _dim1: std::ops::RangeFull) -> &mut [T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		unsafe { self.buf_mut() }.get_mut(span).unwrap()
	}
}

trait SliceMut3D<T> {
	fn slice_mut(&mut self, dim0: usize, dim1: usize, dim2: std::ops::RangeFull) -> &mut [T];
}

impl<T: Copy + HasDType + FromToF64> SliceMut3D<T> for generic::Tensor<ND<3>, &mut [T]> {
	fn slice_mut(&mut self, dim0: usize, dim1: usize, _dim2: std::ops::RangeFull) -> &mut [T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let map = map.select(0, dim1).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		unsafe { self.buf_mut() }.get_mut(span).unwrap()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct ContiguousOutput<'t, T> {
	pub tensor: generic::Tensor<&'t ND<2>, &'t mut [T]>,
}

pub struct ContiguousInput<'t, T> {
	pub tensor: generic::Tensor<&'t ND<2>, &'t [T]>,
}

pub struct BroadcastedInput<'t, T> {
	pub tensor: generic::Tensor<&'t ND<2>, &'t [T]>,
}

pub enum Input<'t, T> {
	Contiguous(ContiguousInput<'t, T>),
	Broadcasted(BroadcastedInput<'t, T>),
}

#[repr(C)]
pub(super) struct CPUFloatVMT<T: Copy + HasDType + FromToF64> {
	vmt: DeviceBufferVMT,
	phantom: std::marker::PhantomData<T>,
}

impl<T: 'static + Copy + HasDType + FromToF64> CPUFloatVMT<T> {
	pub fn new(device: &Rc<MaybeUninit<CPUDevice>>, kernel_runner: Rc<KernelRunner>) -> Self {
		let device = device.as_ptr();
		let device = unsafe { NonNull::new_unchecked(device as *mut CPUDevice) };
		let device_is_cpu = true;
		let dtype = T::dtype;
		Self {
			vmt: unsafe {
				DeviceBufferVMT::new(
					device,
					device_is_cpu,
					dtype,
					kernel_runner,
					CPUDevice::drop_buffer,
					Self::read_float,
					Self::read_bin,
					Self::write_bin,
					Self::mm,
					Self::attention,
					Self::run_kernel,
				)
			},
			phantom: std::marker::PhantomData,
		}
	}

	unsafe fn cast_this<'a>(vmt: NonNull<DeviceBufferVMT>) -> &'a Self {
		debug_assert!(std::mem::offset_of!(Self, vmt) == 0);
		let vmt = vmt.cast::<Self>();
		unsafe { &*vmt.as_ptr() }
	}

	fn device(&self) -> &CPUDevice {
		let (device, _) = self.vmt.device_ptr().to_raw_parts();
		let cpu_device = device.cast::<CPUDevice>();
		unsafe { cpu_device.as_ref() }
	}

	pub fn view_contiguous<'t, 'buf>(
		tensor: &'t generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<ContiguousInput<'t, T>, ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		if !feature_dim.is_contiguous() {
			return Err(ExecutorError::not_contiguous());
		}
		Ok(ContiguousInput { tensor: tensor.view()? })
	}

	pub fn view_contiguous_mut<'t, 'buf>(
		tensor: &'t mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<ContiguousOutput<'t, T>, ErrPack<ExecutorError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		if !feature_dim.is_contiguous() {
			return Err(ExecutorError::not_contiguous());
		}
		Ok(ContiguousOutput { tensor: tensor.view_mut()? })
	}

	pub fn attention_tile<const FIRST: bool>(
		acc: &mut generic::Tensor<ND<3>, &mut [f64]>, // [output, head, vo_feature]

		q: &generic::Tensor<ND<3>, &[T]>, // [output, head, qk_feature]
		k: &generic::Tensor<ND<3>, &[T]>, // [input, head, qk_feature]
		v: &generic::Tensor<ND<3>, &[T]>, // [input, head, vo_feature]

		// `acc`, `prev_max` and `prev_sum` will be initialized when processing the first tile.
		prev_max: &mut generic::Tensor<ND<2>, &mut [f64]>, // [output, head]
		prev_sum: &mut generic::Tensor<ND<2>, &mut [f64]>, // [output, head]

		// Scratch space for storing scores. It doesn't need to be initialized.
		// On GPU, its shape will be [output, head, input]. However, we process outputs
		// sequentially, so we don't need separate space for each output.
		scores: &mut generic::Tensor<ND<2>, &mut [f64]>, // [head, input]
	) {
		let O = q.size(0).unwrap();
		let I = k.size(0).unwrap();
		let H = q.size(1).unwrap();
		let VO = v.size(2).unwrap();
		for j in 0..O {
			for i in 0..I {
				for h in 0..H {
					let q = q.slice(j, h, ..);
					let k = k.slice(i, h, ..);
					scores[[h, i]] = math::dot(q, k);
					// scores[h][i].set(math::dot(q, k))
				}
			}
			if FIRST {
				for h in 0..H {
					let prev_max = &mut prev_max[[j, h]];
					let prev_sum = &mut prev_sum[[j, h]];

					let scores = scores.slice_mut(h, ..);
					let scores = &mut scores[..I]; // TODO
					let (first_max, first_sum) = math::softmax_part1_(scores);

					*prev_max = first_max;
					*prev_sum = first_sum;

					let acc = acc.slice_mut(j, h, ..);
					for i in 0..1 {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64();
						for f in 0..VO {
							acc[f] = score * v[f].to_f64();
						}
					}
					for i in 1..I {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64();
						for f in 0..VO {
							acc[f] += score * v[f].to_f64();
						}
					}
				}
			} else {
				for h in 0..H {
					let prev_max = &mut prev_max[[j, h]];
					let prev_sum = &mut prev_sum[[j, h]];

					let scores = scores.slice_mut(h, ..);
					let scores = &mut scores[..I]; // TODO
					let (new_max, new_sum) = math::softmax_part1_(scores);

					let total_max = prev_max.max(new_max);
					*prev_max = total_max;

					let prev_weight = (*prev_max - total_max).exp();
					let new_weight = (new_max - total_max).exp();

					*prev_sum = (*prev_sum * prev_weight) + (new_sum * new_weight);

					let acc = acc.slice_mut(j, h, ..);
					for i in 0..1 {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64() * new_weight;
						for f in 0..VO {
							acc[f] = (acc[f] * prev_weight) + (score * v[f].to_f64());
						}
					}
					for i in 1..I {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64() * new_weight;
						for f in 0..VO {
							acc[f] += score * v[f].to_f64();
						}
					}
				}
			}
		}
	}

	// TODO - Extract this function into `DynamicEvalExpr` struct that will keep
	// elemwise_args, reduce_args, scalar_args, reduction_size.
	// If we don't have to pass these around, all the recursive calls will be just one line
	// and this function will be much shorter.
	//
	// This way we will get rid of both clippy warnings.
	#[allow(clippy::too_many_lines)]
	#[allow(clippy::too_many_arguments)]
	pub unsafe fn eval_expr(
		&self,
		expr: &DynExpr,
		j: usize,
		i: usize,
		k: usize,
		elemwise_args: &[KernelElemArg],
		reduce_args: &[KernelReduceArg],
		scalar_args: &[f64],
		reduction_size: usize,
	) -> f64 {
		unsafe {
			match expr {
				DynExpr::ElemwiseTensorArg(index) => {
					assert!(k == 0);
					let elemwise_arg = &elemwise_args[*index];
					elemwise_arg
						.device_data
						.add(
							elemwise_arg.offset_bytes
								+ j * elemwise_arg.stride_bytes[0]
								+ i * elemwise_arg.stride_bytes[1],
						)
						.cast::<T>()
						.read()
						.to_f64()
				},
				DynExpr::ReduceTensorArg(index) => {
					let reduce_arg = &reduce_args[*index];
					assert!(k < reduction_size);
					reduce_arg
						.device_data
						.add(
							reduce_arg.offset_bytes
								+ j * reduce_arg.stride_bytes[0]
								+ i * reduce_arg.stride_bytes[1]
								+ k * reduce_arg.stride_bytes[2],
						)
						.cast::<T>()
						.read()
						.to_f64()
				},
				DynExpr::ScalarArg(index) => scalar_args[*index],

				DynExpr::RandnExpr() => {
					let mut rng = self.device().rng.borrow_mut();
					let val = rng.get_normal_clamped();
					val
				},

				DynExpr::SumExpr(a) => {
					assert!(!reduce_args.is_empty());
					let a = a.as_ref();
					let mut sum = 0.0;
					for k in 0..reduction_size {
						let value = self.eval_expr(
							a,
							j,
							i,
							k,
							elemwise_args,
							reduce_args,
							scalar_args,
							reduction_size,
						);
						sum += value;
					}
					sum
				},
				DynExpr::MaxExpr(a) => {
					assert!(!reduce_args.is_empty());
					let a = a.as_ref();
					(0..reduction_size)
						.map(|k| {
							self.eval_expr(
								a,
								j,
								i,
								k,
								elemwise_args,
								reduce_args,
								scalar_args,
								reduction_size,
							)
						})
						.fold(f64::NEG_INFINITY, f64::max)
				},

				DynExpr::ExpExpr(a) => {
					let a = self.eval_expr(
						a,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					a.exp()
				},
				DynExpr::AbsExpr(a) => {
					let a = self.eval_expr(
						a,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					a.abs()
				},
				DynExpr::SigmoidExpr(a) => {
					let a = self.eval_expr(
						a,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					math::sigmoid(a)
				},
				DynExpr::SwishExpr(a) => {
					let a = self.eval_expr(
						a,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					math::swish(a)
				},
				DynExpr::SqrtExpr(a) => {
					let a = self.eval_expr(
						a,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					a.sqrt()
				},
				DynExpr::LnClampedExpr(a) => {
					let a = self.eval_expr(
						a,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					a.ln().max(-1000.0)
				},
				DynExpr::AddExpr(a, b) => {
					let a = self.eval_expr(
						a,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					let b = self.eval_expr(
						b,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					a + b
				},
				DynExpr::MulExpr(a, b) => {
					let a = self.eval_expr(
						a,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					let b = self.eval_expr(
						b,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					a * b
				},
				DynExpr::RecipExpr(a, eps) => {
					let a = self.eval_expr(
						a,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					let eps = self.eval_expr(
						eps,
						j,
						i,
						k,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					);
					1.0 / (a + eps)
				},
			}
		}
	}

	fn read_float<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		src: &generic::Tensor<ND<0>, DeviceBufferRef<'buf>>,
	) -> Result<f64, ErrPack<ExecutorError>> {
		src.ensure_safe()?;
		let view = src.view::<T>()?;
		Ok(view[[]].to_f64())
	}

	fn read_bin<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		dst: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		src: &mut dyn std::io::Read,
	) -> Result<(), ErrPack<ExecutorError>> {
		for q in dst.iter_along_axis(0) {
			//q.ensure_safe()?;
		}
		let (map, buf) = Self::view_contiguous_mut(dst)?.tensor.into_parts();
		for j in 0..map.dims[0].size {
			let b = map.index_to_offset([j, 0]).unwrap();
			let e = b + map.dims[1].size;
			let buf = &mut buf[b..e];
			let byte_slice = unsafe {
				std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), size_of_val(buf))
			};
			src.read_exact(byte_slice)?;
		}
		Ok(())
	}

	fn write_bin<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		_src: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_dst: &mut dyn std::io::Write,
	) -> Result<(), ErrPack<ExecutorError>> {
		/*
		#[cfg(target_endian = "big")]
		{
			todo!("Saving to binary file on big-endian targets is not implemented yet");
		}
		let mut result = Ok(());
		let src = Self::view_contiguous(src)?;
		unsafe {
			zip_vecs([], [src], |[], [src]| {
				if result.is_ok() {
					let ptr = src.as_ptr().cast();
					let bytes = std::mem::size_of_val(src);
					let slice = std::slice::from_raw_parts(ptr, bytes);
					result = dst.write_all(slice);
				}
			});
		}
		result.map_err(Into::into)
		*/
		todo!("Saving to binary file is not implemented yet");
	}

	#[allow(clippy::panic_in_result_fn)]
	#[allow(clippy::many_single_char_names)]
	fn mm<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		let m = o.map().dims[0].size;
		let n = o.map().dims[1].size;
		let k = a.map().dims[1].size;

		assert!(a.map().dims[0].size == m);
		assert!(b.map().dims[0].size == k);
		assert!(b.map().dims[1].size == n);

		o.ensure_safe()?;
		let o_row_stride = o.map().dims[0].stride;
		let o_col_stride = o.map().dims[1].stride;
		let mut o = o.view_mut::<T>()?;
		let o_off = o.map().offset;
		let o = unsafe { &mut o.buf_mut()[o_off..] };

		a.ensure_safe()?;
		let a_row_stride = a.map().dims[0].stride;
		let a_col_stride = a.map().dims[1].stride;
		let a = a.view::<T>()?;
		let a = &a.buf()[a.map().offset..];

		b.ensure_safe()?;
		let b_row_stride = b.map().dims[0].stride;
		let b_col_stride = b.map().dims[1].stride;
		let b = b.view::<T>()?;
		let b = &b.buf()[b.map().offset..];

		for j in 0..m {
			for i in 0..n {
				let mut t = 0.0;
				for k in 0..k {
					let a = a[j * a_row_stride + k * a_col_stride];
					let b = b[k * b_row_stride + i * b_col_stride];
					t += a.to_f64() * b.to_f64();
				}
				let t = T::from_f64(t * scale);
				o[j * o_row_stride + i * o_col_stride] = t;
			}
		}

		Ok(()) // TODO
	}

	fn attention(
		_this: NonNull<DeviceBufferVMT>,
		_o: &mut generic::Tensor<ND<3>, DeviceBufferRefMut>, // [inputs, qo_heads, vo_features]
		_q: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, qo_heads, qk_features]
		_k: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, k_heads, qk_features]
		_v: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, v_heads, vo_features]
	) {
		todo!("CPUFloatExecutor::attention is not implemented yet");
	}

	unsafe fn run_kernel(
		this: NonNull<DeviceBufferVMT>,
		kernel_data: &KernelData,
		o: *const KernelOutput,
		elemwise_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		scalar_args: *const f64,
		reduction_size: usize,
	) -> Result<(), ErrPack<ExecutorError>> {
		let this = unsafe { Self::cast_this(this) };
		let expr = kernel_data.expr.as_ref();
		unsafe {
			let elemwise_args =
				std::slice::from_raw_parts(elemwise_args, kernel_data.elemwise_count);
			let reduce_args = std::slice::from_raw_parts(reduce_args, kernel_data.reduce_count);
			let scalar_args = std::slice::from_raw_parts(scalar_args, kernel_data.scalar_count);
			let o = &*o;
			for j in 0..o.size[0] {
				for i in 0..o.size[1] {
					let o = o
						.device_data
						.add(o.offset_bytes + j * o.stride_bytes[0] + i * o.stride_bytes[1])
						.cast::<T>();
					o.write(T::from_f64(this.eval_expr(
						expr,
						j,
						i,
						0,
						elemwise_args,
						reduce_args,
						scalar_args,
						reduction_size,
					)));
				}
			}
		}
		Ok(())
	}
}
