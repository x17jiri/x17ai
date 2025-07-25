# Copyright 2025 Jiri Bobek. All rights reserved.
# License: GPL 3.0 or later. See LICENSE.txt for details.
#
#---------------------------------------------------------------------------------------------------

import sys
import torch

def gen_softmax():
	inp = torch.tensor([
		[-1.2719, -0.6884, -0.6477, -1.3343, -1.7648],
		[-1.9440,  0.9989,  2.8260, -0.3503, -0.5406],
		[ 0.1619, -0.9744, -0.6539,  1.9764,  0.7423],
		[ 0.0689,  1.1983,  0.0077, -0.6580, -0.4917],
	], dtype=torch.float32, requires_grad=True)

	out = torch.nn.functional.softmax(inp, dim=-1)

	print("expected_out = ", out)

	d_out = torch.tensor([
		[ 0.1000,  0.2000, -0.3000, -0.1000,  0.7000],
		[ 0.0500, -0.1500,  0.1000,  0.0000,  0.6500],
		[-0.2000,  0.1000,  0.0500,  0.0500,  0.3331],
		[ 0.0000,  0.1000, -0.0500, -0.0500, -0.1442],
	], dtype=torch.float32)

	out.backward(d_out)

	print("expected_d_inp = ", inp.grad)

def gen_rms_norm():
	inp = torch.tensor([
		[-1.2719, -0.6884, -0.6477, -1.3343, -1.7648],
		[-1.9440,  0.9989,  2.8260, -0.3503, -0.5406],
		[ 0.1619, -0.9744, -0.6539,  1.9764,  0.7423],
		[ 0.0689,  1.1983,  0.0077, -0.6580, -0.4917],
	], dtype=torch.float32, requires_grad=True)

	out = torch.nn.functional.rms_norm(inp, normalized_shape=(5,), eps=1e-5)

	print("expected_out = ", out)

	d_out = torch.tensor([
		[ 0.1000,  0.2000, -0.3000, -0.1000,  0.7000],
		[ 0.0500, -0.1500,  0.1000,  0.0000,  0.6500],
		[-0.2000,  0.1000,  0.0500,  0.0500,  0.3331],
		[ 0.0000,  0.1000, -0.0500, -0.0500, -0.1442],
	], dtype=torch.float32)

	out.backward(d_out)

	print("expected_d_inp = ", inp.grad)

def gen_skip_con():
	inp = torch.tensor([
		[-1.2719, -0.6884, -0.6477, -1.3343, -1.7648],
		[-1.9440,  0.9989,  2.8260, -0.3503, -0.5406],
		[ 0.1619, -0.9744, -0.6539,  1.9764,  0.7423],
		[ 0.0689,  1.1983,  0.0077, -0.6580, -0.4917],
	], dtype=torch.float32, requires_grad=True)

	out = torch.nn.functional.rms_norm(inp, normalized_shape=(5,), eps=1e-5) + inp

	print("expected_out = ", out)

	d_out = torch.tensor([
		[ 0.1000,  0.2000, -0.3000, -0.1000,  0.7000],
		[ 0.0500, -0.1500,  0.1000,  0.0000,  0.6500],
		[-0.2000,  0.1000,  0.0500,  0.0500,  0.3331],
		[ 0.0000,  0.1000, -0.0500, -0.0500, -0.1442],
	], dtype=torch.float32)

	out.backward(d_out)

	print("expected_d_inp = ", inp.grad)

def gen_swiglu():
	input = torch.tensor([
		[
			[-0.8714,  0.0940, -1.8542,  0.7304, -0.2773, -0.7002, -0.6732],
			[-0.3639,  0.3168, -0.1291,  0.0461,  1.1104,  1.1302,  0.9531]
		], [
			[-0.4135,  0.8453, -1.5096, -0.6099,  0.3261,  0.0221,  0.2651],
			[-0.6350,  1.2983, -0.3186,  1.1820,  0.8960,  0.3618,  0.7403]
		], [
			[ 1.4432,  3.0449,  1.6251, -0.9864,  0.5130,  0.7087,  0.0929],
			[-1.0867, -0.9725,  0.6218,  0.0897, -0.6929,  0.6524,  2.0153]
		], [
			[-1.2271, -1.5085,  0.6593,  1.9288, -1.3475, -2.3681,  1.6462],
			[ 1.1650, -0.5926,  0.2599,  1.1419, -1.1015,  0.7637,  1.9136]
		],
	], dtype=torch.float32, requires_grad=True)

	linear = input[:, 0, :]
	gate = input[:, 1, :]

	out = linear * torch.nn.functional.silu(gate)

	print("expected_out = ", out)

	d_out = torch.tensor([
		[-0.8866,  0.5923,  1.6628, -0.0801, -0.8071, -0.4731, -0.5233],
		[ 1.1934, -0.0170,  0.1537,  0.4730, -0.5441,  0.9822,  0.1070],
		[-1.4121, -0.9289,  0.4226,  0.2391, -1.1131, -0.2210, -0.7231],
		[ 0.4270,  1.0933,  0.2516, -0.1986,  0.7602, -1.7255, -0.9384],
	], dtype=torch.float32)

	out.backward(d_out)

	print("expected_d_inp = ", input.grad)

def gen_linear():
	weights = torch.tensor([
		[-0.7392, -0.4243, -2.2199, -0.7662, -0.4344],
		[-1.1176, -0.5131,  0.5884,  1.6860, -0.5456],
		[-0.6692, -0.7482, -0.5937, -0.4305, -1.6972],
		[ 1.0881, -0.7972, -1.2000, -0.6788, -0.9008],
		[ 1.8882, -1.1999,  0.3821, -0.2152,  0.2094],
		[-1.1796, -1.8167,  1.2314, -0.6760,  0.0761],
	], dtype=torch.float32, requires_grad=True)

	input = torch.tensor([
		[-1.2794, -0.1038,  0.3636, -0.0918, -0.6903],
		[-0.2495, -1.7407, -0.4136,  1.2375,  0.0408],
		[ 0.2060, -1.0269,  0.2663,  1.8425,  1.4105],
		[-1.8738,  1.0913,  0.5786, -0.8210,  0.0362],
	], dtype=torch.float32, requires_grad=True)

	out = torch.matmul(weights, input.T).T
	forward_scale = (1.0 / (weights.shape[1] ** 0.5))
	backward_scale = 1.0
	scaled_out = out * forward_scale

	print("expected_out = ", scaled_out)

	d_out = torch.tensor([
		[-1.2192,  0.9470, -1.0698,  1.0365,  0.1644, -0.1481],
		[-1.0424, -0.4814, -1.5834,  0.4658,  1.0362, -0.2995],
		[-0.5644,  1.4450, -1.0186, -0.5245,  2.2684, -0.5567],
		[-0.9963, -1.7835,  0.6185,  2.0077, -0.5136, -0.9927]
	], dtype=torch.float32)

	out.backward(d_out)

	d_inp = input.grad * backward_scale

	print("expected_d_inp = ", d_inp);

	d_weights = weights.grad

	print("expected_d_weights = ", d_weights)

def gen_wrapper():
	# torch.randn(4, 4) * 5
	weights = torch.tensor([
		[ 0.4410,  1.0766,  3.7616, -6.4873],
		[ 9.6167,  1.4782, -1.6646, -9.1857],
		[ 1.6518,  1.8148,  0.7772, -3.1704],
		[-3.5822,  5.1029,  0.5198, -2.1559],
	], dtype=torch.float32, requires_grad=True)
	#weights = torch.randn(4, 4) * 2
	#weights.requires_grad = True

	# torch.randn(5, 4) * 3
	input = torch.tensor([
		[-2.7016,  2.2977, -1.3401,  0.8465],
		[ 0.1577,  1.2480,  2.9994, -1.7373],
		[-1.9980,  3.0446,  5.2757, -1.2850],
		[-2.1982, -0.8580, -0.6265, -1.6002],
		[-2.2695, -1.1619, -3.1626,  2.6974],
	], dtype=torch.float32, requires_grad=True)

	rms = (input*input).mean(dim=-1, keepdim=True).sqrt()
	norm = input / (rms + 1e-5)
	print("norm = ", norm)

	mm_inp = norm
	mm_inp.retain_grad()

	forward_scale = (1.0 / (weights.shape[1] ** 0.5))
	out = torch.matmul(weights, mm_inp.T).T * forward_scale

	rms_out = (out*out).mean(dim=-1, keepdim=True).sqrt()
	ratio = rms_out / (rms + 1e-5)

	final_out = out + input

	print("expected_out = ", final_out)
	print("expected_ratio = ", ratio)

	# torch.randn(5, 4) * 0.5
	d_out = torch.tensor([
		[-0.3847,  0.0664, -0.6217,  0.5983],
		[ 0.2775,  0.2007, -0.7973,  0.0167],
		[ 0.5394, -0.0854,  0.5947, -0.4340],
		[ 0.0528, -0.1087,  0.2953,  0.3368],
		[ 0.1521, -0.4394, -0.6771, -0.0617],
	], dtype=torch.float32)

	d_out_magn = (d_out*d_out).mean(dim=-1, keepdim=True).sqrt()
	print("d_out_magn = ", d_out_magn)

	final_out.backward(d_out)

	d_mm = mm_inp.grad
	d_inp_real = input.grad
	d_inp_real_magn = (d_inp_real*d_inp_real).mean(dim=-1, keepdim=True).sqrt()
	d_weights = weights.grad

	rms_d_out = (d_out*d_out).mean(dim=-1, keepdim=True).sqrt()
	rms_d_mm = (d_mm*d_mm).mean(dim=-1, keepdim=True).sqrt()
	ratio_d = rms_d_mm / (rms_d_out + 1e-5)
	d_inp = d_mm / ratio_d * ratio + d_out
	d_inp_magn = (d_inp*d_inp).mean(dim=-1, keepdim=True).sqrt()

	d_inp_rescaled = d_inp / (d_inp_magn + 1e-5) * d_out_magn
	print("d_inp_rescaled = ", d_inp_rescaled)

	print("expected_d_mm = ", d_mm)
	print("expected_d_inp = ", d_inp)
	print("expected_d_inp_magn = ", d_inp_magn)
	print("expected_d_inp2 = ", d_inp / (d_inp_magn + 1e-5))
	print("expected_d_inp_real = ", d_inp_real)
	print("expected_d_inp_real_magn = ", d_inp_real_magn)
	print("expected_d_inp_real2 = ", d_inp_real / (d_inp_real_magn + 1e-5))
	print("expected_d_weights = ", d_weights / forward_scale)

what = ''
try:
	what = sys.argv[1]
except:
	pass

options = {
	'softmax': gen_softmax,
	'rms_norm': gen_rms_norm,
	'skip_con': gen_skip_con,
	'swiglu': gen_swiglu,
	'linear': gen_linear,
	'wrapper': gen_wrapper,
}

if __name__ == "__main__":
	if what in options:
		options[what]()
	else:
		sys.stderr.write("Usage:\n")
		sys.stderr.write("\tpython gen_test_data.py <WHAT>\n")
		sys.stderr.write("Available options for <WHAT>:\n")
		for key in sorted(options.keys()):
			sys.stderr.write(f"\t{key}\n")
		sys.exit(1)
