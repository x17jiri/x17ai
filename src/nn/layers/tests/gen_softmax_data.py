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

if __name__ == "__main__":
	gen_softmax_data()
