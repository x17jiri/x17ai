import numpy as np
import torch
import sys

shape = (4*4096, 192)

def load_bf16_from_file(path, shape):
    # Read raw 16‑bit data
    arr_u16 = np.fromfile(path, dtype=np.uint16)
    arr_u16 = arr_u16.reshape(shape)

    # Make a torch tensor with same bits and view as bfloat16
    t_u16 = torch.from_numpy(arr_u16)
    t_bf16 = t_u16.view(torch.bfloat16)  # reinterpret bits
    return t_bf16

def create_random_bf16_to_file(path, shape):
    # Generate random float32 data
    arr_f32 = np.random.randn(*shape).astype(np.float32)

    # Convert to torch tensor and cast to bfloat16
    t_f32 = torch.from_numpy(arr_f32)
    t_bf16 = t_f32.to(torch.bfloat16)

    # View as uint16 to get raw bits
    t_u16 = t_bf16.view(torch.uint16)

    # Convert back to numpy and save
    arr_u16 = t_u16.numpy()
    arr_u16.tofile(path)

create_random_bf16_to_file("q.bin", shape)
create_random_bf16_to_file("kv.bin", shape)

q  = load_bf16_from_file("q.bin",  shape).to(torch.float32)
kv = load_bf16_from_file("kv.bin", shape).to(torch.float32)

scores = torch.matmul(q, kv.transpose(-2, -1))

#print("scores:", scores[:16,:16])
#print("kv 0:", kv[:16,:16])
#print("kv 1:", kv[16:32,:16])

o = torch.matmul(scores.to(torch.bfloat16).to(torch.float32), kv[:, :128]).to(torch.bfloat16).to(torch.float32)

print("pytorch:", o[1568:1568+16,64:64+16])

test_output = load_bf16_from_file("out_cpu.bin", (4*4096, 128)).to(torch.float32)

print("my code:", test_output[1568:1568+16,64:64+16])

thr = 1e-4
mask = (torch.abs(o) > thr) | (torch.abs(test_output) > thr)
o_filtered = torch.where(mask, o, torch.zeros_like(o))
t_filtered = torch.where(mask, test_output, torch.zeros_like(test_output))
abs_diff = torch.abs(o_filtered - t_filtered)
relative_error = abs_diff / (torch.abs(o_filtered).min(torch.abs(t_filtered)) + 1e-8)
max_relative_error = torch.max(relative_error)
max_location = torch.where(relative_error == max_relative_error)
print("relative_error:", max_relative_error)
print("max_location:", max_location)

print("out at max: ", o[1573, 75])
print("tst at max: ", test_output[1573, 75])

#print("out at max: ", o[622, 26])
#print("tst at max: ", test_output[622, 26])

#print("out at max: ", o[622, 67])
#print("tst at max: ", test_output[622, 67])

sys.exit(0) ########################################################################################

# calculate simple attention (not casual and not scaled)
#scores = torch.nn.functional.softmax(scores, dim=-1)

scores_max = torch.max(scores, dim=-1, keepdim=True).values
scores_exp = torch.exp(scores - scores_max).to(torch.bfloat16).to(torch.float64)
scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
scores = scores_exp / scores_sum

v = kv[:, :128]
output = torch.matmul(scores, v)

print("pytorch:", output[3802])

test_output = load_bf16_from_file("out_cpu.bin", (4096, 128)).to(torch.float64)
print("my code:", test_output[3802])

# Calculate max difference
o = output#[:256,:]
thr = 1e-3
mask = (torch.abs(o) > thr) | (torch.abs(test_output) > thr)
o_filtered = torch.where(mask, o, torch.zeros_like(o))
t_filtered = torch.where(mask, test_output, torch.zeros_like(test_output))
abs_diff = torch.abs(o_filtered - t_filtered)
relative_error = abs_diff / (torch.abs(o_filtered).min(torch.abs(t_filtered)) + 1e-8)
max_relative_error = torch.max(relative_error)
max_location = torch.where(relative_error == max_relative_error)
print("relative_error:", max_relative_error)
print("max_location:", max_location)

print("out at max: ", o[2873, 69])
print("tst at max: ", test_output[2873, 69])

print("out at max: ", o[622, 26])
print("tst at max: ", test_output[622, 26])

print("out at max: ", o[622, 67])
print("tst at max: ", test_output[622, 67])

#print("q:", q[0])
#print("kv:", kv[0])
#print("dot:", q[0]*kv[0])
#for i in range(192):
#	print("b:", kv[0][i]);
#	print("a:",  q[1][i]);
#	print("score:", i+1, " -> ", (q[1][:i+1]*kv[0][:i+1]).sum())
#print("scores after softmax:", scores)
