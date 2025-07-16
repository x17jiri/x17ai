//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::sync::Arc;

use crate::ErrPack;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut, check_borrows};
use crate::tensor::device::kernel_registry::KernelRegistry;
use crate::tensor::dim_merger::{DimMerger, MergedDim};
use crate::tensor::generic::map::ND;
use crate::tensor::{Tensor, TensorOpError, generic};
use crate::util::array;

//--------------------------------------------------------------------------------------------------

pub enum ScalarExpr {
	ElemArg(Arc<ElemArg>),
	ConstArg(Arc<ConstArg>),
	FloatLiteral(FloatLiteral),

	DotExpr(Arc<VecArg>, Arc<VecArg>),

	AddExpr(Arc<ScalarExpr>, Arc<ScalarExpr>),
}

pub struct ScalarExprWrapper {
	pub expr: Arc<ScalarExpr>,
}

pub struct VecArgWrapper {
	pub arg: Arc<VecArg>,
}

//--------------------------------------------------------------------------------------------------

pub struct ElemArg {
	pub index: usize,
	pub name: String,
}

pub struct ConstArg {
	pub index: usize,
	pub name: String,
}

pub struct FloatLiteral {
	pub value: f64,
}

pub struct VecArg {
	pub index: usize,
	pub name: String,
}

pub struct KernelData {
	pub id: usize,
	pub name: String,
	pub elem_args: Box<[Arc<ElemArg>]>,
	pub vec_args: Box<[Arc<VecArg>]>,
	pub const_args: Box<[Arc<ConstArg>]>,
	pub expr: Arc<ScalarExpr>,
}

pub struct Kernel<const E: usize, const V: usize, const C: usize> {
	data: Arc<KernelData>,
}

impl<const E: usize, const V: usize, const C: usize> Kernel<E, V, C> {
	fn new(data: Arc<KernelData>) -> Self {
		debug_assert!(data.elem_args.len() == E);
		debug_assert!(data.vec_args.len() == V);
		debug_assert!(data.const_args.len() == C);
		Self { data }
	}

	pub fn call(
		&self,
		output: &Tensor,
		elem_args: [&Tensor; E],
		vec_args: [&Tensor; V],
		const_args: [f64; C],
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); 1 + E + V]:,
	{
		let output_dims = output.map().dims.as_slice();
		let elem_dims = elem_args.map(|t| t.map().dims.as_slice());
		let Some(vec_dims) = vec_args.try_map(|t| t.map().dims.as_slice().split_last()) else {
			cold_path();
			return Err(TensorOpError::missing_vec_dimension());
		};
		let vec_features = vec_dims.map(|(&feature_dim, _)| feature_dim);
		let vec_dims = vec_dims.map(|(_, dim)| dim);
		let all_dims = array::concat_arrays([output_dims], elem_dims);
		let all_dims = array::concat_arrays(all_dims, vec_dims);

		let merged: [MergedDim<{ 1 + E + V }>; 2] = DimMerger::merge(all_dims)?;

		unsafe {
			let mut c_fail = 0;
			let elem_tensors: [generic::Tensor<ND<2>, DeviceBufferRef>; E] =
				std::array::from_fn(|i| {
					generic::Tensor::new_unchecked(
						ND {
							dims: [merged[0].get(1 + i), merged[1].get(1 + i)],
							offset: elem_args[i].map().offset,
						},
						DeviceBufferRef::new_unsafe(elem_args[i].buf().as_ref(), &mut c_fail),
					)
				});
			let vec_tensors: [generic::Tensor<ND<3>, DeviceBufferRef>; V] =
				std::array::from_fn(|i| {
					generic::Tensor::new_unchecked(
						ND {
							dims: [
								merged[0].get(1 + E + i),
								merged[1].get(1 + E + i),
								vec_features[i],
							],
							offset: vec_args[i].map().offset,
						},
						DeviceBufferRef::new_unsafe(vec_args[i].buf().as_ref(), &mut c_fail),
					)
				});
			let mut fail = c_fail;
			let mut o_tensor = generic::Tensor::new_unchecked(
				ND {
					dims: [merged[0].get(0), merged[1].get(0)],
					offset: output.map().offset,
				},
				DeviceBufferRefMut::new_unsafe(output.buf().as_ref(), &mut fail),
			);
			check_borrows(c_fail, fail)?;

			let executor = output.executor();

			executor.run_kernel(
				self.data.as_ref(),
				&mut o_tensor,
				&elem_tensors,
				&vec_tensors,
				&const_args,
			)?;
		}
		Ok(())
	}
}

pub struct KernelBuilder<const E: usize, const V: usize, const C: usize> {
	pub(crate) name: String,
	pub(crate) elem_args: [Arc<ElemArg>; E],
	pub(crate) vec_args: [Arc<VecArg>; V],
	pub(crate) const_args: [Arc<ConstArg>; C],
}

impl<const E: usize, const V: usize, const C: usize> KernelBuilder<E, V, C> {
	pub fn new(
		name: &str,
		elem_args: [&str; E],
		vec_args: [&str; V],
		const_args: [&str; C],
	) -> (Self, [ScalarExprWrapper; E], [VecArgWrapper; V], [ScalarExprWrapper; C]) {
		let elem_args = array::map_into(elem_args, |index, name| {
			Arc::new(ElemArg { index, name: name.to_string() })
		});
		let vec_args = array::map_into(vec_args, |index, name| {
			Arc::new(VecArg { index, name: name.to_string() })
		});
		let const_args = array::map_into(const_args, |index, name| {
			Arc::new(ConstArg { index, name: name.to_string() })
		});

		let builder = Self {
			name: name.to_string(),
			elem_args: elem_args.clone(),
			vec_args: vec_args.clone(),
			const_args: const_args.clone(),
		};

		let elem_args_exprs =
			elem_args.map(|a| ScalarExprWrapper { expr: Arc::new(ScalarExpr::ElemArg(a)) });
		let vec_args_exprs = vec_args.map(|a| VecArgWrapper { arg: a });
		let const_args_exprs =
			const_args.map(|a| ScalarExprWrapper { expr: Arc::new(ScalarExpr::ConstArg(a)) });

		(builder, elem_args_exprs, vec_args_exprs, const_args_exprs)
	}

	pub fn build(self, expr: ScalarExprWrapper) -> Kernel<E, V, C> {
		let ScalarExprWrapper { expr } = expr;
		let Self { name, elem_args, vec_args, const_args } = self;

		let elem_args = elem_args.into();
		let vec_args = vec_args.into();
		let const_args = const_args.into();

		let reg = KernelRegistry::instance();
		let mut reg = reg.write().unwrap();
		let data = reg.add_kernel(|id| {
			Arc::new(KernelData {
				id,
				name,
				elem_args,
				vec_args,
				const_args,
				expr,
			})
		});
		Kernel::new(data)
	}
}

//--------------------------------------------------------------------------------------------------

impl std::ops::Add<Self> for ScalarExprWrapper {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self {
			expr: Arc::new(ScalarExpr::AddExpr(self.expr, rhs.expr)),
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct VecMul(Arc<VecArg>, Arc<VecArg>);

impl VecMul {
	pub fn sum(self) -> ScalarExprWrapper {
		ScalarExprWrapper {
			expr: Arc::new(ScalarExpr::DotExpr(self.0, self.1)),
		}
	}
}

impl std::ops::Mul<Self> for VecArgWrapper {
	type Output = VecMul;

	fn mul(self, rhs: Self) -> Self::Output {
		VecMul(self.arg, rhs.arg)
	}
}

//--------------------------------------------------------------------------------------------------
