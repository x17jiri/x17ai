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
	input =

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
