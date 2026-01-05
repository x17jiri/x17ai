//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#![allow(warnings)] // TODO - disabling warnings for main.rs. Remove this later.
#![allow(non_snake_case)]
#![feature(generic_const_exprs)]
#![feature(macro_metavar_expr)]
#![feature(string_from_utf8_lossy_owned)]
#![feature(f16)]
#![feature(thin_box)]

//use x17ai::nn::layers::{Layer, Linear, LossFunction, SoftmaxCrossEntropy};
//use x17ai::nn::{EvalContext, ModelContext};
//use x17ai::tensor::device::cpu::CPUDevice;
//use x17ai::tensor::math::Savable;
//use x17ai::{debug_2d, tensor};

/*
struct Attention {
	pub input_features: usize,
	pub heads: usize,
	pub qk_size: usize,
	pub v_size: usize,
	pub dtype: DType,

	pub k: Linear,
	pub q: Linear,
	pub v: Linear,
}

impl Attention {
	pub fn new(
		input_features: usize,
		heads: usize,
		qk_size: usize,
		v_size: usize,
		dtype: DType,
		alloc: &mut dyn Allocator,
	) -> Attention {
		let k = Linear::new(input_features, qk_size, dtype, alloc);
		let q = Linear::new(input_features, heads * qk_size, dtype, alloc);
		let v = Linear::new(input_features, heads * v_size, dtype, alloc);

		Attention {
			input_features,
			heads,
			qk_size,
			v_size,
			dtype,
			k,
			q,
			v,
		}
	}
}

impl Module for Attention {
	// input is of the form: [..., inputs, embeding]
	fn forward(&self, input: &Tensor) -> Tensor {
		// explanation of dimension names:
		// *: batch (can be any number of dimensions >= 0)
		// i: input sequence
		// h: head
		// q, k, v: key, query, value

		// TODO - use scopes so tensors are freed when not needed

		// input: [*, i, input_features]
		let seq_len = input.shape()[-2];

		// k: [*, i, k]
		// -> [*, 1, i, k]
		// -> [*, h, i, k]
		let k = self.k.forward(input);
		let k = k.reshape_last_n(2, &[1, seq_len, self.qk_size]);

		// q: [*, i, h * q]
		// -> [*, i, h, q]
		// -> [*, h, i, q]
		// -> [*, h, q, i]
		let q = self.q.forward(input);
		let q = q.reshape_last_n(1, &[self.heads, self.qk_size]);
		let q = q.transposed(-3, -2);
		let q = q.transposed(-2, -1);

		// v: [*, i, h * v]
		// -> [*, i, h, v]
		// -> [*, h, i, v]
		let v = self.v.forward(input);
		let v = v.reshape_last_n(1, &[self.heads, self.v_size]);
		let w_shape = v.shape().to_vec(); // [*, i, h, v]
		let v = v.transposed(-3, -2);

		// scores: [*, h, i, i]
		let scores = matmul(&k, &q);

		// w = reweighted v
		// w: [*, h, i, v]
		// -> [*, i, h, v]
		// -> [*, i, w = h * v]

		let w = v.new_tensor(&w_shape, v.dtype()); // [*, i, h, v]
		let w = w.transposed(-3, -2); // [*, h, i, v]

		matmul_(scores, v, w);
		let w = w.transposed(-3, -2); // [*, i, h, v]
		let w = w.reshape_last_n(2, &[self.heads * self.v_size]);

		w
	}
}

struct Transformer {
	pub attention: Attention,
	pub feed_forward: Linear,
}

impl Transformer {
	pub fn new(
		input_features: usize,
		heads: usize,
		qk_size: usize,
		v_size: usize,
		dtype: DType,
		alloc: &mut dyn Allocator,
	) -> Transformer {
		let attention = Attention::new(input_features, heads, qk_size, v_size, dtype, alloc);
		let feed_forward = Linear::new(heads * v_size, 2 * input_features, dtype, alloc);
		Transformer { attention, feed_forward }
	}
}

impl Module for Transformer {
	fn output_info(&self, input: &Tensor) -> (Rc<Shape>, DType) {
		(input.shape.clone(), input.dtype)
	}

	fn forward_(&self, input: &Tensor, output: &Tensor, _ctx: &Context) {
		let a = self.rms_norm.forward(input);
		let b = self.attention.forward(a);
		let c = self.feed_forward.forward(b);
		swiglu_(c, output);
	}
}
*/

/*use ndarray::{Array2, ArrayView2, s};

#[allow(clippy::reversed_empty_ranges)]
fn laplacian(v: &ArrayView2<f32>) -> Array2<f32> {
	let q = s![..-2, 1..-1];

	-4. * &v.slice(s![1..-1, 1..-1])
		+ v.slice(s![..-2, 1..-1])
		+ v.slice(s![1..-1, ..-2])
		+ v.slice(s![1..-1, 2..])
		+ v.slice(s![2.., 1..-1])
}*/

use std::boxed::ThinBox;
use std::rc::Rc;

use thin_vec::thin_vec;
use x17ai::autograd::{AutogradTensor, LossFn};
use x17ai::new::autograd::{Autograd, AutogradExpr, BackwardFn};
use x17ai::new::expr::compile2::PreCompilation;
use x17ai::new::expr::{CanBeBatched, Expr, ScalarRef, TensorRef};
use x17ai::new::nn::linear::Linear;
use x17ai::new::nn::rms_norm::{RMSNormGrad, rms_norm};
use x17ai::new::nn::swiglu::swiglu;
use x17ai::new::tensor::TensorLiteral1D;
use x17ai::nn::ModelContext;
use x17ai::nn::fragments::softmax::{Softmax, SoftmaxGradMode};
use x17ai::nn::fragments::{CrossEntropy, Fragment, UnaryFragment};
use x17ai::rng::Rng;
use x17ai::tensor::device::cpu::CPUDevice;
use x17ai::tensor::device::cuda::CudaDevice;
use x17ai::tensor::device::dtype::common_dtype;
use x17ai::tensor::math::{col, mat, row};
use x17ai::tensor::{DType, Device, HasDType, Tensor, TensorOpError};
use x17ai::{ErrPack, custom_kernel, tensor};

#[cfg(false)]
fn main() -> Result<(), ErrPack<TensorOpError>> {
	let dev = CPUDevice::new();
	let lit = GenericTensor::literal_factory::<f32>(dev.clone());
	let a = lit.new_2d(&[
		[-1.5924, 0.7530, -0.2418, 0.3416],
		[0.1225, -1.1488, 0.3338, -0.8102],
		[-0.3426, 1.7670, 0.0596, -0.1160],
	])?;
	let b = lit.new_1d(&[
		-0.5254, //
		2.2503,  //
		-0.2887, //
		-0.2313, //
	])?;
	let c = GenericTensor::new_empty_on(&[3], f32::dtype, dev.clone())?;
	col(&c)?.assign(mat(&a)? * col(&b)?)?;

	let a = lit.new_1d(&[
		-1.2854, //
		0.3690,  //
		0.0352,  //
		0.4591,  //
		-0.2684, //
	])?;
	let b = lit.new_1d(&[0.3259, -2.2469, 0.8345, 0.6012])?;
	let c = GenericTensor::new_empty_on(&[5, 4], f32::dtype, dev.clone())?;
	mat(&c)?.clear_acc(col(&a)? * row(&b)?)?;

	println!("a = {}", a.borrow()?.view::<f32>()?);
	println!("b = {}", b.borrow()?.view::<f32>()?);
	println!("c = {}", c.borrow()?.view::<f32>()?);

	/*	let (map, elems) = ND::new(&[3, 4, 5])?;
	let buf = dev.clone().new_buffer(f32::dtype, elems)?;*/
	/*	let t = Tensor { map, buf: buf.as_ref() };
	let exec = &dev.f32_executor;
	exec.mm(&t, &t, &t, 1.0)?;*/

	Ok(())
}

/*
fn test1_opt(dev: Rc<dyn x17ai::new::device::Device>) -> Expr {
	let my_lit = TensorLiteral1D::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0]);

	let mut v_ten = TensorRef::new(Some("v".into()), f32::dtype, vec![1]);
	let mut m_ten = TensorRef::new(Some("m".into()), f32::dtype, vec![]);
	let mut grad_ten = TensorRef::new(Some("grad".into()), f32::dtype, vec![]);
	let mut value_ten = TensorRef::new(Some("value".into()), f32::dtype, vec![]);

	let grad1 = Expr::new_tensor_input(grad_ten.clone());
	let grad2 = Expr::new_tensor_input(grad_ten.clone());
	let grad3 = Expr::new_tensor_input(grad_ten.clone());

	let m = Expr::new_tensor_input(m_ten.clone());
	//let m = m.capture(ExprTensorRef::new(Some("m_captured".into()), f32::dtype, vec![]));
	let m_decay_coef =
		Expr::new_scalar_input(ScalarRef::new(Some("m_decay_coef".into()), f32::dtype));
	let m_update_coef =
		Expr::new_scalar_input(ScalarRef::new(Some("m_update_coef".into()), f32::dtype));

	let m_decayed = (m * m_decay_coef); //.max(); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	let m_update = grad1.clone() * m_update_coef;
	let new_m = m_decayed + m_update;
	let new_m = new_m.capture(m_ten.clone());

	let v = Expr::new_tensor_input(v_ten.clone());
	let v_decay_coef =
		Expr::new_scalar_input(ScalarRef::new(Some("v_decay_coef".into()), f32::dtype));
	let v_update_coef =
		Expr::new_scalar_input(ScalarRef::new(Some("v_update_coef".into()), f32::dtype));

	let v_decayed = v.clone() * v_decay_coef;
	let v_update = (grad2.clone() * grad3).sum() * v_update_coef.clone();
	//let v_update = v_update.sum() * v_update_coef;
	let new_v = v_decayed + v_update;
	let new_v = new_v.capture(v_ten.clone());
	//let new_v = new_v.capture(value_ten.clone());

	let eps = Expr::new_scalar_input(ScalarRef::new(Some("eps".into()), f32::dtype));
	let v_rsqrt = (new_v.sqrt() + eps).recip();
	//return v_rsqrt;

	let value = Expr::new_tensor_input(value_ten.clone());
	let update_coef =
		Expr::new_scalar_input(ScalarRef::new(Some("update_coef".into()), f32::dtype));
	let update = new_m.clone() * v_rsqrt;
	let new_value = value + update.clone() * update_coef;
	let new_value = new_value.capture(value_ten.clone());
	let new_value = new_value.clone() + update.sqrt();

	v_ten.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[500, 1], f32::dtype, dev.clone()).unwrap(),
	);
	m_ten.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[500, 300], f32::dtype, dev.clone()).unwrap(),
	);
	grad_ten.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[500, 300], f32::dtype, dev.clone()).unwrap(),
	);
	value_ten.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[500, 300], f32::dtype, dev.clone()).unwrap(),
	);

	new_value

	/*
	let x_ten = ExprTensorRef::new(Some("x".into()), f32::dtype, vec![]);

	RcExpr::first(new_value.clone(), (new_value + new_m).capture(x_ten))*/
}
*/

fn x_rms_norm(
	inp: Expr,
	eps: Rc<ScalarRef>,
	internal_dtype: DType,
) -> Result<Expr, ErrPack<TensorOpError>> {
	let eps = Expr::new_scalar_input(eps.clone());
	let inp = inp.cast(internal_dtype);

	let magn_recip = ((inp.clone() * inp.clone()).mean().sqrt() + eps).recip().cast(f32::dtype);

	Ok(inp * magn_recip)
}

fn x_softmax(inp: Expr, internal_dtype: DType) -> Result<Expr, ErrPack<TensorOpError>> {
	let inp = inp.cast(internal_dtype);

	let max = inp.clone().max();
	let t = (inp - max).exp();
	let out = t.clone().sum().recip() * t;

	Ok(out)
}

pub fn x_swiglu(inp: Expr, internal_dtype: DType) -> Result<Expr, ErrPack<TensorOpError>> {
	let inp = inp.cast(internal_dtype);
	let lin = inp.clone().select_even();
	let gate = inp.select_odd();

	let one = Expr::new_const("ONE", 1.0);

	let out = lin * gate.clone() * ((-gate).exp() + one).recip();

	Ok(out)
}

/*
fn test2_rms_norm(dev: Rc<dyn x17ai::new::device::Device>) -> Expr {
	let inp_ten = TensorRef::new(Some("inp".into()), f32::dtype, Vec::new());
	let inp2_ten = TensorRef::new(Some("inp2".into()), f32::dtype, Vec::new());
	let inp = Expr::new_tensor_input(inp_ten.clone());
	let inp2 = Expr::new_tensor_input(inp_ten.clone());
	let sum_to_mean = ScalarRef::new(Some("sum_to_mean".into()), f32::dtype);
	let sum_to_mean = Expr::new_scalar_input(sum_to_mean.clone());
	let eps = ScalarRef::new(Some("eps".into()), f32::dtype);
	let eps = Expr::new_scalar_input(eps.clone());
	let magn_recip = (((inp.clone() * inp2.clone()).sum() * sum_to_mean).sqrt() + eps).recip();

	inp_ten.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[2, 1000, 512], f32::dtype, dev.clone()).unwrap(),
	);

	inp2_ten.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[1, 1000, 512], f32::dtype, dev.clone()).unwrap(),
	);

	(inp * magn_recip).capture(inp_ten.clone()).capture(inp2_ten.clone())
}

fn test3_softmax(dev: Rc<dyn x17ai::new::device::Device>) -> Expr {
	let inp_ten = TensorRef::new(Some("inp".into()), f32::dtype, vec![15, 32]);
	let inp = Expr::new_tensor_input(inp_ten.clone());
	let max = inp.clone().max();
	let t = (inp - max).exp();
	let out = t.clone().sum().recip() * t;
	let out = out;
	inp_ten.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[1000, 512], f32::dtype, dev.clone()).unwrap(),
	);
	let sca = ScalarRef::new(Some("m_decay_coef".into()), f32::dtype);
	let coef1 = Expr::new_scalar_input(sca.clone());
	let coef2 = Expr::new_scalar_input(sca.clone());

	((out.clone() + coef1).capture(inp_ten.clone())
		- (coef2 + out.clone()).capture(inp_ten.clone()))
}

fn test4_x(dev: Rc<dyn x17ai::new::device::Device>) -> Expr {
	let mut inp_a = TensorRef::new(Some("a".into()), f32::dtype, vec![]);
	let mut inb_b = TensorRef::new(Some("b".into()), f32::dtype, vec![]);
	let mut inp_c = TensorRef::new(Some("c".into()), f32::dtype, vec![]);
	let out = TensorRef::new(Some("out".into()), f32::dtype, vec![]);
	let a = Expr::new_tensor_input(inp_a.clone());
	let b = -Expr::new_tensor_input(inb_b.clone());
	let c = Expr::new_tensor_input(inp_c.clone());
	let max = (a + b.clone()).max();
	let sum = (b + c).sum();
	let out = (max * sum).capture(out);

	inp_a.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[1000, 512], f32::dtype, dev.clone()).unwrap(),
	);
	inb_b.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[1000, 512], f32::dtype, dev.clone()).unwrap(),
	);
	inp_c.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[1, 512], f32::dtype, dev.clone()).unwrap(),
	);

	out
}

fn test5(dev: Rc<dyn x17ai::new::device::Device>) -> Expr {
	let mut inp_a = TensorRef::new(Some("a".into()), f32::dtype, vec![]);
	let mut inp_b = TensorRef::new(Some("b".into()), f32::dtype, vec![]);
	let a = Expr::new_tensor_input(inp_a.clone());
	let b = Expr::new_tensor_input(inp_b.clone());
	let out = TensorRef::new(Some("out".into()), f32::dtype, vec![]);
	let out = (a + b).capture(out);

	inp_a.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[1000, 512], f32::dtype, dev.clone()).unwrap(),
	);
	inp_b.tensor.borrow_mut().replace(
		x17ai::new::tensor::Tensor::new_empty(&[1000, 1], f32::dtype, dev.clone()).unwrap(),
	);

	out
}
*/

/*
pub struct ModuleTemplate {
	//
}

pub fn transformer() -> Module {
	//
}

pub fn model() {
	let model = Model::new();
	let t = transformer();
	let layer1 = model.add_module("layer1", t.clone());
	layer1.set_param("mq", mq);
	layer1.set_param("mkv", mkv);
	layer1.set_param("mw", mw);

	let layer2 = model.add_module("layer2", t.clone());
	layer2.set_param("mq", mq);
	layer2.set_param("mkv", mkv);
	layer2.set_param("mw", mw);

	model.connect(model.input(), layer1["input"]);
	model.connect(layer1["output"], layer2["input"]);
	model.connect(layer2["output"], model.output());
}
*/

pub struct FakeBackwardFn;

impl BackwardFn for FakeBackwardFn {
	fn run(self: Box<Self>, d_out: Expr, autograd: &mut Autograd) {
		autograd.eval(d_out);
	}
}

fn main() -> Result<(), ErrPack<TensorOpError>> {
	let dev = x17ai::new::device::cpu::CPUDevice::new();

	let io_dtype = f16::dtype;
	let internal_dtype = f32::dtype;
	let inp = TensorRef::new("inp", io_dtype, &[1024], CanBeBatched::Yes);
	let out = TensorRef::new("out", io_dtype, &[1024], CanBeBatched::Yes);
	let fake_backward_fn = Box::new(FakeBackwardFn);
	let fwd = rms_norm(
		AutogradExpr::new(inp, Some(fake_backward_fn)),
		0.001,
		internal_dtype,
		RMSNormGrad::Precise,
	);
	let lin = Linear::new("lin", 1024, 2048, io_dtype);
	let fwd = lin.forward(fwd, internal_dtype);
	let fwd = swiglu(fwd, internal_dtype);
	let fwd_captured = fwd.expr.capture_into(out);

	let d_inp = TensorRef::new("d_inp", io_dtype, &[1024], CanBeBatched::Yes);
	let d_out = TensorRef::new("d_out", io_dtype, &[1024], CanBeBatched::Yes);

	let mut comp = PreCompilation::new(&fwd_captured.node);
	let graphviz = comp.print_graphviz();
	std::fs::write("graph.dot", graphviz).unwrap();
	let graphviz = comp.print_fragment_graphviz();
	std::fs::write("fragments.dot", graphviz).unwrap();

	let bwd = Autograd::run(fwd.backward_fn, d_out);
	let bwd = bwd.capture_into(d_inp.clone());

	let mut comp = PreCompilation::new(&bwd.node);
	let graphviz = comp.print_graphviz();
	std::fs::write("graph2.dot", graphviz).unwrap();
	let graphviz = comp.print_fragment_graphviz();
	std::fs::write("fragments2.dot", graphviz).unwrap();

	return Ok(());

	let eps = ScalarRef::new("eps");

	//let expr = test1_opt(dev.clone());
	//let expr = test2_rms_norm(dev.clone());
	//let expr = test3_softmax(dev.clone());
	//let expr = test4_x(dev.clone());
	//let expr = test5(dev.clone());

	//let q = TensorRef::new("q".into(), io_dtype, vec![4, 4, 64], CanBeBatched::Yes);
	//let kv = TensorRef::new("kv".into(), io_dtype, vec![1, 4, 64 + 64], CanBeBatched::Yes);

	let t = TensorRef::new("t", io_dtype, &[1024], CanBeBatched::Yes);
	let r = TensorRef::new("r", io_dtype, &[1024], CanBeBatched::Yes);
	let mq = TensorRef::new("mq", io_dtype, &[1024, 1024], CanBeBatched::No);
	let mkv = TensorRef::new("mkv", io_dtype, &[1024, 4 * (64 + 64)], CanBeBatched::No);
	let mw = TensorRef::new("mw", io_dtype, &[1024, 2048], CanBeBatched::No);
	let expr = Expr::new_tensor_input(t.clone());

	let expr = AutogradExpr::new(expr, Some(Box::new(FakeBackwardFn)));
	let expr = rms_norm(expr, 0.001, internal_dtype, RMSNormGrad::Precise);
	let expr = expr.expr;
	//let expr = x_rms_norm(expr, eps.clone(), internal_dtype)?.cast(io_dtype);

	let qq = expr
		.clone()
		.cast(internal_dtype)
		.row_times_mat(Expr::new_tensor_input(mq.clone()).cast(internal_dtype))
		.cast(io_dtype)
		.reshape(&[4, 4, 64])
		.cast(internal_dtype);
	//.capture(q.clone());
	let kv = expr
		.clone()
		.cast(internal_dtype)
		.row_times_mat(Expr::new_tensor_input(mkv.clone()).cast(internal_dtype))
		.cast(io_dtype)
		.cast(internal_dtype)
		.reshape(&[1, 4, 64 + 64]);
	//.capture(kv.clone())
	//.capture(q.clone());

	//let expr = RcExpr::first(q, kv);
	//let expr = x_softmax(expr, internal_dtype)?;
	//	let expr = x_swiglu(expr, internal_dtype)?;

	let expr = qq.attention(kv);
	let expr = expr
		.cast(io_dtype)
		.cast(internal_dtype)
		.reshape(&[1024])
		.row_times_mat(Expr::new_tensor_input(mw.clone()).cast(internal_dtype));
	let expr = x_swiglu(expr, internal_dtype)?;
	let expr = expr.cast(io_dtype);
	//let expr = x_rms_norm(expr, eps.clone(), internal_dtype)?.cast(io_dtype);
	let expr = expr.capture_into(r.clone());

	let mut comp = PreCompilation::new(&expr.node);

	let graphviz = comp.print_graphviz();
	std::fs::write("graph.dot", graphviz).unwrap();

	let graphviz = comp.print_fragment_graphviz();
	std::fs::write("fragments.dot", graphviz).unwrap();

	/*let mut comp = CompiledExpr::new(comp);
	let dev = x17ai::new::device::cpu::CPUDevice::new();
	let inp_lit = TensorLiteral1D::<f32>::new(&[1.0, 2.0, 3.0]);
	let inp_ten = x17ai::new::tensor::Tensor::new(dev, &inp_lit)?;*/
	//inp.tensor.borrow_mut().replace(inp_ten);
	//comp.frag_shapes();
	/*
	let inp = RcExpr::new_tensor_input(f32::dtype, "inp".into());
	let max = inp.clone().max();
	let max = max.capture(ExprTensorRef::new(Some("max".into()), f64::dtype));
	let t = (inp - max).exp();
	let out = t.clone().sum().recip() * t;
	let cap = out.capture(ExprTensorRef::new(Some("out".into()), f64::dtype));
	let mut nodes = NodeVec::new_from_expr(cap.as_ref());
	calc_shape_groups(&mut nodes);
	let mut graphviz = String::new();
	print_graphviz(&mut graphviz, &nodes);
	println!("{}", graphviz);
	*/
	return Ok(());

	/*
	let cpu = CPUDevice::new();
	let dev = CudaDevice::new(0)?;

	let mut a = [1, 2, 3, 4, 5, 6, 7, 8];
	let b = a.get_mut(2..5);

	stderrlog::new().verbosity(10).init().unwrap();
	let mut rng = Rng::default();

	//let cuda_dev = cuda::CUDADevice::new().unwrap();

	let dev = CPUDevice::new();
	let internal_dtype = f32::dtype;
	let momentum_dtype = f32::dtype;
	let lit = Tensor::literal_factory::<f32>(dev.clone());
	let mut mctx = ModelContext::new(dev.clone());

	let mut lin1 = Linear::new(3, 5, f32::dtype, internal_dtype, &mut mctx)?;
	let mut sf = Softmax::new(internal_dtype, SoftmaxGradMode::Precise);
	let mut lin2 = Linear::new(5, 2, f32::dtype, internal_dtype, &mut mctx)?;
	lin1.randomize(&mut rng)?;
	lin2.randomize(&mut rng)?;
	mctx.init_optimizer(momentum_dtype, internal_dtype)?;

	let loss_layer = CrossEntropy::new(internal_dtype);

	let input = Tensor::new_empty_on(&[2, 3], f32::dtype, dev.clone())?;
	input.randn_(&mut rng)?;

	let expected = lit.new_2d(&[[1.0, 0.0], [0.0, 1.0]])?;

	//	for (name, param) in model.named_params("model_params") {
	//		println!("{}: {}", name, param.borrow().value().borrow()?.view::<f32>()?);
	//	}

	for _ in 0..10000 {
		//		println!("Step {}", i);
		//		println!();

		let a = AutogradTensor::new(input.clone(), None);
		let b = lin1.forward(a)?;
		let c = sf.forward(b)?;
		let d = lin2.forward(c)?;
		//println!("d = {}", d.value.borrow()?.view::<f32>()?);
		let loss_fn = loss_layer.forward_with_target(d, expected.clone())?;
		let loss = loss_fn.loss()?;

		mctx.zero_grad();

		let grad_dtype = f32::dtype;
		loss_fn.backward(grad_dtype)?;

		mctx.step()?;

		println!("{}", &loss);
		//println!("--------------------------------------------------");
	}

	//	for (name, param) in model.named_params("model_params") {
	//		println!("{}: {}", name, param.borrow().value().borrow()?.view::<f32>()?);
	//	}
	println!("input = {}", &input);
	println!("expected = {}", &expected);
	*/

	/*	let t = Tensor::new_empty_on(&[5, 7], f32::dtype, dev.clone());
	let _ = t.read_from_file("tensor.bin");
	println!("t = {}", t);*/

	/*
	//	let Q = Tensor::new_empty_on(&[10, 8, 64], f32::dtype, dev.clone());
	let Q = Tensor::new_empty_on(&[10, 4, 5], f32::dtype, dev.clone());
	let K = Q.new_empty_like();
	let V = Q.new_empty_like();
	let expected_out = Q.new_empty_like();
	let out = Q.new_empty_like();

	Q.read_file("Q.bin").unwrap();
	K.read_file("K.bin").unwrap();
	V.read_file("V.bin").unwrap();
	expected_out.read_file("expected_out.bin").unwrap();

	Q.write_file("QQ.bin").unwrap();

	let rank0 = Tensor::new_empty_on(&[], f32::dtype, dev.clone());
	rank0.read_file("rank0.bin").unwrap();

	tensor::math::attention(&Q, &K, &V).save_to(&out);
	let mut out_txt = std::fs::File::create("out.txt").unwrap();
	writeln!(out_txt, "out = {}", out).unwrap();
	*/
	/*
	Q = tensor([[[-0.5728,  0.2759, -1.2481,  1.2678,  0.0717],
			 [ 0.7365, -0.1365, -1.0994, -0.5166,  1.5378],
			 [ 0.2821,  0.2301,  0.1800, -0.0684,  0.8570],
			 [-0.1845, -0.5259, -0.9884,  0.3163, -0.5252]],

			[[ 1.2729, -1.8302,  0.1313, -0.0683, -0.5589],
			 [ 0.0100, -0.3041,  0.2271,  1.0116,  1.0592],
			 [-0.3654,  2.6270,  0.8530,  1.2593,  0.0454],
			 [ 0.5415,  1.2081,  1.5722,  1.3396,  0.2734]],

			[[-0.2628, -0.0934,  1.1173,  0.2901,  0.3048],
			 [ 0.9385, -1.4884,  0.8204, -1.7372,  2.7728],
			 [ 0.0775, -0.7416, -0.2225, -0.0360, -1.8122],
			 [ 1.0723, -0.7707, -0.8560, -1.4648,  1.2018]]])
	K = tensor([[[-0.4923,  0.7813,  0.8644, -1.5706,  0.7498],
			 [-1.4921, -3.1150, -0.9135, -0.7144,  1.2236],
			 [ 2.0291,  0.0276,  1.0505, -1.9432, -1.3391],
			 [-0.4148, -1.1546, -0.3438,  0.9069, -0.5321]],

			[[ 1.0127, -0.1844, -1.0132,  2.0210,  0.8130],
			 [-0.5455, -0.5098, -1.9764,  0.3509, -1.4468],
			 [ 0.0681, -1.0504,  0.6592,  0.1989, -1.4774],
			 [-0.3097,  0.6771,  0.2947, -1.4067, -1.4417]],

			[[ 0.2734, -0.4254,  1.1729,  2.0100, -1.1407],
			 [-0.2713, -0.1622, -0.5532,  0.9638, -0.9287],
			 [ 0.7248, -0.7776, -0.1271,  1.5814, -0.4532],
			 [ 1.5086, -0.0872,  0.5254, -0.2733,  0.5933]]])
	V = tensor([[[ 2.6798, -0.5179,  1.2582, -0.3989, -0.2235],
			 [-1.3648, -0.4122,  1.0682, -0.8368, -1.1953],
			 [ 0.7335,  0.3082, -0.6078, -1.4065, -0.6577],
			 [ 0.3178, -0.8711,  1.2750, -0.3165,  0.2562]],

			[[ 0.1106, -1.0367, -2.2284, -0.4045, -1.2343],
			 [-0.3686,  0.4288, -0.3380, -0.1972,  0.3778],
			 [-0.3289,  0.2250, -0.1146,  1.0722, -0.1600],
			 [ 0.6813, -1.2130, -1.1754,  0.1417,  0.2431]],

			[[ 0.2859, -1.3778,  0.6605, -1.2354,  1.1392],
			 [-1.9311, -0.2242, -2.3424,  0.1420, -1.0550],
			 [ 0.2309, -0.5817,  1.5256, -2.0075, -0.6199],
			 [-1.7653,  2.8820, -0.2060, -1.4014,  0.2813]]])
	scores= [
		[
			[-2.518785, 3.254305, 0.728666]
			[-2.254767, 0.901010, 1.780698]
			[0.795129, -0.546805, 1.513646]
		],
		[
			[2.581363, -0.565546, -1.495607]
			[1.298100, -1.476732, -0.087805]
			[7.120363, -5.996059, -4.716572]
		],
		[
			[-0.246985, -1.383708, -0.494012]
			[-2.280602, -2.038418, -0.445138]
			[2.399786, 3.307809, 1.425397]
		],
		[
			[1.589838, -0.277994, -1.149758]
			[-1.090536, -1.165001, 1.333642]
			[-1.228544, -0.778282, 2.348536]
		],
	]
		*/
	Ok(())
}
