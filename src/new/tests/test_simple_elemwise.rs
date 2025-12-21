//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------
/*
use crate::ErrPack;
use crate::new::device::cpu::CPUDevice;
use crate::new::device::{Device, KernelArgs};
use crate::new::expr::compile::FragmentIndex;
use crate::new::expr::{ExprTensorRef, RcExpr};
use crate::new::tensor::{Tensor, TensorLiteral1D};
use crate::tensor::{HasDType, TensorOpError};
*/
//--------------------------------------------------------------------------------------------------
/*
#[test]
fn test_simple_elemwise() -> Result<(), ErrPack<TensorOpError>> {
	let dev = CPUDevice::new();
	let a = Tensor::new(dev.clone(), &TensorLiteral1D::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0]));
	let b = Tensor::new(dev.clone(), &TensorLiteral1D::<f32>::new(&[10.0, 20.0, 30.0, 40.0, 50.0]));

	let a_ref = ExprTensorRef::new(Some("a".into()), f32::dtype, vec![]);
	let b_ref = ExprTensorRef::new(Some("b".into()), f32::dtype, vec![]);
	let c_ref = ExprTensorRef::new(Some("c".into()), f32::dtype, vec![]);

	let a = RcExpr::new_tensor_input(a_ref.clone());
	let b = RcExpr::new_tensor_input(b_ref.clone());

	let expr = (a + b).capture(c_ref.clone());

	let comp = expr.compile();
	let fragments = comp.fragments();
	assert_eq!(fragments.len(), 1);
	let frag_index = FragmentIndex::new(0);
	let frag = &fragments[frag_index];
	let inp_count = frag.tensor_inputs_vec.len();
	let out_count = frag.tensor_outputs_vec.len();
	let scalar_count = frag.scalar_inputs_vec.len();
	assert_eq!(inp_count, 2);
	assert_eq!(out_count, 1);
	assert_eq!(scalar_count, 0);

	let mut args = KernelArgs::new(KernelArgs::extra_memory(inp_count, out_count, scalar_count));
	args.set_counts(inp_count, out_count, scalar_count)?;

	let inputs = args.inputs_mut();
	let outputs = args.outputs_mut();
	let scalars = args.scalars_mut();

	//unsafe { dev.run_fragment(&comp, frag_index, &args)? };

	Ok(())
}
*/
//--------------------------------------------------------------------------------------------------
