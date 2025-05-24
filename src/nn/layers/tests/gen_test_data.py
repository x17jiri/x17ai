import sys
import torch

def gen_softmax_data():
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

def gen_rms_norm_data():
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

def gen_skip_con_data():
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

def gen_swiglu_data():
	input = torch.tensor([
		[-0.0640,  1.1649, -0.4438, -0.9085,  2.6784, -0.5343, -0.6572],
		[ 0.2617,  0.0550,  0.1319,  0.3918, -0.0658, -0.7212,  0.6670],

		[ 0.0378,  0.8686, -0.1275, -1.0481,  0.4591, -0.5995,  1.2170],
		[-2.8136, -0.9275, -0.9066, -1.3998,  1.5730, -2.0097,  0.1782],

		[ 1.3420,  0.2817, -0.5869, -0.6039,  0.0472,  0.7144, -1.0679],
		[ 0.0147,  0.4431, -0.8810,  0.9642,  0.0969,  0.1036, -1.1162],

		[ 0.1500, -1.2500, -0.0651,  1.3316, -0.9012, -1.6888, -0.1247],
		[-0.0266,  0.1781, -0.2857,  0.4415,  1.1776,  1.6097, -0.3190],
	], dtype=torch.float32, requires_grad=True)
	inputs = inputs.reshape(4, 2, 7)

	lin = inputs[:, 0, :]
	gate = inputs[:, 1, :]

	out = linear * torch.nn.functional.silu(gate)

	print("expected_out = ", out)

	d_out = torch.tensor([
		[ 0.7395,  0.6013,  0.2484,  0.2852, -0.5568, -0.3432,  0.7511],
		[-0.8545, -1.5350, -1.7655,  0.4452, -0.3133, -0.3896, -0.1574],
		[ 2.1220,  0.8120,  1.5263, -0.8563, -0.2650, -0.8958,  0.4047],
	], dtype=torch.float32)

	out.backward(d_out)

	print("expected_d_linear = ", linear.grad)
	print("expected_d_gate = ", gate.grad)

what = ''
try:
	what = sys.argv[1]
except:
	pass

options = {
	'softmax': gen_softmax_data,
	'rms_norm': gen_rms_norm_data,
	'skip_con': gen_skip_con_data,
	'swiglu': gen_swiglu_data,
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
