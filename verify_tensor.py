#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import torch


PCT_BUCKET_MIN_MAGNITUDES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
PCT_BUCKET_LABELS = "1e-6,     1e-5,     1e-4,     1e-3,     1e-2,     1e-1"
WORST_PCT_REPORT_BUCKETS = [(1e-1, "1e-1"), (1e-2, "1e-2"), (1e-3, "1e-3")]


def load_bf16(path: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
	raw = Path(path).read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16)
	if shape is not None:
		return data.reshape(shape)
	return data


def load_f32(path: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
	raw = Path(path).read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.float32)
	if shape is not None:
		return data.reshape(shape)
	return data


def load_f32_bits(path: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
	raw = Path(path).read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.int32)
	if shape is not None:
		return data.reshape(shape)
	return data


def f8_e4m3fn_lut() -> torch.Tensor:
	values: list[float] = []
	for code in range(256):
		sign = -1.0 if (code & 0x80) else 1.0
		exp = (code >> 3) & 0x0F
		mant = code & 0x07
		if exp == 0 and mant == 0:
			values.append(math.copysign(0.0, sign))
		elif exp == 0:
			values.append(sign * math.ldexp(mant / 8.0, -6))
		elif exp == 0x0F and mant == 0x07:
			values.append(math.nan)
		else:
			values.append(sign * math.ldexp(1.0 + mant / 8.0, exp - 7))
	return torch.tensor(values, dtype=torch.float32)


def load_f8_bits(path: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
	raw = Path(path).read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
	if shape is not None:
		return data.reshape(shape)
	return data


def load_f8(path: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
	bits = load_f8_bits(path, shape)
	return f8_e4m3fn_lut()[bits.to(torch.long)]


def infer_dtype_from_paths(*paths: str) -> str:
	if any("_f32" in Path(path).stem for path in paths):
		return "f32"
	if any("_f8" in Path(path).stem for path in paths):
		return "f8"
	return "bf16"


def normalize_bf16_zero_sign(bits: torch.Tensor) -> torch.Tensor:
	normalized = bits.clone()
	zero_mask = (normalized & 0x7FFF) == 0
	normalized[zero_mask] = 0
	return normalized


def bf16_ordered_int(bits: torch.Tensor) -> torch.Tensor:
	bits_i32 = bits.to(torch.int32)
	sign = (bits_i32 >> 15) & 1
	return torch.where(sign == 0, bits_i32 + 0x8000, 0x8000 - (bits_i32 & 0x7FFF))


def f32_ordered_int(bits: torch.Tensor) -> torch.Tensor:
	bits_i64 = bits.to(torch.int64) & 0xFFFFFFFF
	sign = (bits_i64 >> 31) & 1
	return torch.where(sign == 0, bits_i64 + 0x80000000, 0x80000000 - (bits_i64 & 0x7FFFFFFF))


def f8_ordered_int(bits: torch.Tensor) -> torch.Tensor:
	bits_i32 = bits.to(torch.int32)
	sign = (bits_i32 >> 7) & 1
	return torch.where(sign == 0, bits_i32 + 0x80, 0x80 - (bits_i32 & 0x7F))


def infer_shape(file_a: str, file_b: str, shape_args: list[int] | None, bytes_per_elem: int) -> tuple[int, ...]:
	bytes_a = Path(file_a).stat().st_size
	bytes_b = Path(file_b).stat().st_size
	if bytes_a != bytes_b:
		raise ValueError(f"File sizes differ: {file_a}={bytes_a} bytes, {file_b}={bytes_b} bytes")
	if bytes_a % bytes_per_elem != 0:
		raise ValueError(
			f"Expected file size to be divisible by element size {bytes_per_elem} bytes, got {bytes_a}"
		)
	elem_count = bytes_a // bytes_per_elem
	if shape_args is None:
		return (elem_count,)
	shape = tuple(shape_args)
	shape_elems = 1
	for dim in shape:
		shape_elems *= dim
	if shape_elems != elem_count:
		raise ValueError(f"Provided shape {shape} has {shape_elems} elements, file has {elem_count}")
	return shape


def format_index(flat_index: int, shape: tuple[int, ...]) -> str:
	if len(shape) == 1:
		return str(flat_index)
	coords: list[int] = []
	remaining = flat_index
	for dim in reversed(shape):
		coords.append(remaining % dim)
		remaining //= dim
	coords.reverse()
	return ", ".join(str(coord) for coord in coords)


def variance_one_renorm_scales(
	a: torch.Tensor,
	b: torch.Tensor,
	finite_mask: torch.Tensor,
) -> tuple[float, float] | None:
	if not finite_mask.any():
		return None
	a_scale = a[finite_mask].std(unbiased=False).item()
	b_scale = b[finite_mask].std(unbiased=False).item()
	if not math.isfinite(a_scale) or not math.isfinite(b_scale) or a_scale <= 0.0 or b_scale <= 0.0:
		return None
	return a_scale, b_scale


def worst_pct_diff(
	a: torch.Tensor,
	b: torch.Tensor,
	finite_mask: torch.Tensor,
	min_mag: float,
	renorm_scales: tuple[float, float] | None,
) -> tuple[int, float] | None:
	if renorm_scales is None:
		return None
	a_scale, b_scale = renorm_scales
	a_abs = a.abs()
	b_abs = b.abs()
	mask = finite_mask & (a_abs > min_mag * a_scale) & (b_abs > min_mag * b_scale)
	if not mask.any():
		return None
	hi = torch.maximum(a_abs, b_abs)
	lo = torch.minimum(a_abs, b_abs)
	pct_diff = torch.where(mask, (hi / lo - 1.0) * 100.0, torch.full_like(hi, -1.0))
	flat_idx = int(pct_diff.reshape(-1).argmax().item())
	return flat_idx, pct_diff.reshape(-1)[flat_idx].item()


def max_pct_strs(
	a_abs: torch.Tensor,
	b_abs: torch.Tensor,
	finite_mask: torch.Tensor,
	renorm_scales: tuple[float, float] | None,
) -> list[str]:
	pct_strs = []
	if renorm_scales is None:
		return ["n/a"] * len(PCT_BUCKET_MIN_MAGNITUDES)
	a_scale, b_scale = renorm_scales
	for min_mag in PCT_BUCKET_MIN_MAGNITUDES:
		mask = finite_mask & (a_abs > min_mag * a_scale) & (b_abs > min_mag * b_scale)
		if mask.any():
			hi = torch.maximum(a_abs[mask], b_abs[mask])
			lo = torch.minimum(a_abs[mask], b_abs[mask])
			max_pct = ((hi / lo).max().item() - 1.0) * 100.0
			pct_strs.append(f"{max_pct:.4f}%")
		else:
			pct_strs.append("n/a")
	return pct_strs


def print_worst_pct_summary(
	a: torch.Tensor,
	b: torch.Tensor,
	finite_mask: torch.Tensor,
	shape: tuple[int, ...],
	renorm_scales: tuple[float, float] | None,
) -> None:
	for min_mag, min_mag_label in WORST_PCT_REPORT_BUCKETS:
		worst_pct = worst_pct_diff(a, b, finite_mask, min_mag, renorm_scales)
		if worst_pct is None:
			continue
		worst_pct_idx, worst_pct_value = worst_pct
		worst_pct_ref = a.reshape(-1)[worst_pct_idx].item()
		worst_pct_other = b.reshape(-1)[worst_pct_idx].item()
		print(
			f"Worst pct mismatch (variance-1 MIN_MAG > {min_mag_label}) at [{format_index(worst_pct_idx, shape)}]: "
			f"ref={worst_pct_ref:.6e}, other={worst_pct_other:.6e}, pct={worst_pct_value:.9f}%"
		)


def summarize_nonfinite(name: str, tensor: torch.Tensor) -> str:
	nan_count = int(torch.isnan(tensor).sum().item())
	posinf_count = int(torch.isposinf(tensor).sum().item())
	neginf_count = int(torch.isneginf(tensor).sum().item())
	return f"{name}: nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}"


def sign_flip_summary(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float]:
	flip_mask = torch.isfinite(a) & torch.isfinite(b) & (a != 0) & (b != 0) & (torch.signbit(a) != torch.signbit(b))
	flip_count = int(flip_mask.sum().item())
	if flip_count == 0:
		return 0, 0.0
	largest_mag = torch.maximum(a.abs(), b.abs())[flip_mask].max().item()
	return flip_count, largest_mag


def format_ulp_mismatch_example(
	ulp_diff: torch.Tensor,
	ulp: int,
	a: torch.Tensor,
	b: torch.Tensor,
	shape: tuple[int, ...],
) -> str | None:
	remaining_mask = ulp_diff > ulp
	if not remaining_mask.any():
		return None
	masked_ulp = torch.where(remaining_mask, ulp_diff, torch.full_like(ulp_diff, -1))
	flat_idx = int(masked_ulp.reshape(-1).argmax().item())
	remaining_ulp = int(ulp_diff.reshape(-1)[flat_idx].item())
	ref = a.reshape(-1)[flat_idx].item()
	other = b.reshape(-1)[flat_idx].item()
	return (
		f"  example miss [{format_index(flat_idx, shape)}]: "
		f"ref={ref:.6e}, other={other:.6e}, ulp={remaining_ulp}"
	)


def print_ulp_summary(
	ulp_diff: torch.Tensor,
	total: int,
	a: torch.Tensor,
	b: torch.Tensor,
	shape: tuple[int, ...],
	max_ulp: int = 10,
) -> None:
	for ulp in range(1, max_ulp + 1):
		within_ulp = int((ulp_diff <= ulp).sum().item())
		line = f"Within {ulp:2d} ULP:     {within_ulp}/{total} ({100.0 * within_ulp / total:.2f}%)"
		example = format_ulp_mismatch_example(ulp_diff, ulp, a, b, shape)
		if example is not None:
			line += example
		print(line)
		if within_ulp == total:
			break


def verify_bf16(file_a: str, file_b: str, shape: tuple[int, ...]) -> None:
	a_bf16 = load_bf16(file_a, shape)
	b_bf16 = load_bf16(file_b, shape)
	a = a_bf16.float()
	b = b_bf16.float()

	diff = (a - b).abs()
	max_abs_diff = diff.max().item()
	mean_abs_diff = diff.mean().item()
	flat_idx = int(diff.reshape(-1).argmax().item())
	worst_ref = a_bf16.reshape(-1)[flat_idx].float().item()
	worst_cuda = b_bf16.reshape(-1)[flat_idx].float().item()

	a_bits = a_bf16.view(torch.int16).to(torch.int32) & 0xFFFF
	b_bits = b_bf16.view(torch.int16).to(torch.int32) & 0xFFFF
	a_norm = normalize_bf16_zero_sign(a_bits)
	b_norm = normalize_bf16_zero_sign(b_bits)
	a_ord = bf16_ordered_int(a_bits)
	b_ord = bf16_ordered_int(b_bits)
	ulp_diff = (a_ord - b_ord).abs()
	same_value = a == b
	finite_mask = torch.isfinite(a) & torch.isfinite(b)
	bit_mismatch = a_bits != b_bits
	zero_sign_only = bit_mismatch & same_value
	raw_exact_match = int((a_bits == b_bits).sum().item())
	exact_match = int((a_norm == b_norm).sum().item())
	a_abs = a.abs()
	b_abs = b.abs()
	renorm_scales = variance_one_renorm_scales(a, b, finite_mask)
	pct_strs = max_pct_strs(a_abs, b_abs, finite_mask, renorm_scales)
	sign_flip_count, largest_sign_flip_mag = sign_flip_summary(a, b)
	total = a.numel()

	print("\n--- BF16 Tensor Compare ---")
	print(f"A: {file_a}")
	print(f"B: {file_b}")
	print(f"Shape: [{', '.join(str(dim) for dim in shape)}]")
	print(f"Max abs diff:     {max_abs_diff:.6e}")
	print(f"Mean abs diff:    {mean_abs_diff:.6e}")
	print(f"Max pct diff:     {', '.join(pct_strs)}")
	print("  (MIN_MAG after scaling each tensor to variance 1:")
	print(f"               {PCT_BUCKET_LABELS})")
	print(f"Exact bf16 match: {exact_match}/{total} ({100.0 * exact_match / total:.2f}%)")
	print(f"Raw exact bits:   {raw_exact_match}/{total} ({100.0 * raw_exact_match / total:.2f}%)")
	print(f"Signed-zero-only mismatches: {int(zero_sign_only.sum().item())}")
	print(f"Sign-flipped values: {sign_flip_count}, largest magnitude: {largest_sign_flip_mag:.6e}")
	print_ulp_summary(ulp_diff, total, a, b, shape)
	print(summarize_nonfinite("A non-finite", a))
	print(summarize_nonfinite("B non-finite", b))
	if max_abs_diff > 0.0:
		print(f"Worst abs mismatch at [{format_index(flat_idx, shape)}]: ref={worst_ref:.6e}, other={worst_cuda:.6e}")
	print_worst_pct_summary(a, b, finite_mask, shape, renorm_scales)


def verify_f32(file_a: str, file_b: str, shape: tuple[int, ...]) -> None:
	a = load_f32(file_a, shape)
	b = load_f32(file_b, shape)
	a_bits = load_f32_bits(file_a, shape)
	b_bits = load_f32_bits(file_b, shape)

	finite_mask = torch.isfinite(a) & torch.isfinite(b)
	diff = (a - b).abs()
	if finite_mask.any():
		finite_diff = diff[finite_mask]
		max_abs_diff = finite_diff.max().item()
		mean_abs_diff = finite_diff.mean().item()
		masked_diff = torch.where(finite_mask, diff, torch.full_like(diff, -1.0))
		flat_idx = int(masked_diff.reshape(-1).argmax().item())
		worst_ref = a.reshape(-1)[flat_idx].item()
		worst_other = b.reshape(-1)[flat_idx].item()
	else:
		max_abs_diff = math.nan
		mean_abs_diff = math.nan
		flat_idx = None
		worst_ref = math.nan
		worst_other = math.nan

	a_ord = f32_ordered_int(a_bits)
	b_ord = f32_ordered_int(b_bits)
	ulp_diff = (a_ord - b_ord).abs()
	same_value = a == b
	bit_mismatch = a_bits != b_bits
	zero_sign_only = bit_mismatch & same_value
	raw_exact_match = int((a_bits == b_bits).sum().item())
	exact_match = int(same_value.sum().item())
	a_abs = a.abs()
	b_abs = b.abs()
	renorm_scales = variance_one_renorm_scales(a, b, finite_mask)
	pct_strs = max_pct_strs(a_abs, b_abs, finite_mask, renorm_scales)
	sign_flip_count, largest_sign_flip_mag = sign_flip_summary(a, b)
	total = a.numel()

	print("\n--- F32 Tensor Compare ---")
	print(f"A: {file_a}")
	print(f"B: {file_b}")
	print(f"Shape: [{', '.join(str(dim) for dim in shape)}]")
	print(f"Max abs diff:     {max_abs_diff:.6e}")
	print(f"Mean abs diff:    {mean_abs_diff:.6e}")
	print(f"Max pct diff:     {', '.join(pct_strs)}")
	print("  (MIN_MAG after scaling each tensor to variance 1:")
	print(f"               {PCT_BUCKET_LABELS})")
	print(f"Exact f32 match:  {exact_match}/{total} ({100.0 * exact_match / total:.2f}%)")
	print(f"Raw exact bits:   {raw_exact_match}/{total} ({100.0 * raw_exact_match / total:.2f}%)")
	print(f"Signed-zero-only mismatches: {int(zero_sign_only.sum().item())}")
	print(f"Sign-flipped values: {sign_flip_count}, largest magnitude: {largest_sign_flip_mag:.6e}")
	print_ulp_summary(ulp_diff, total, a, b, shape)
	print(summarize_nonfinite("A non-finite", a))
	print(summarize_nonfinite("B non-finite", b))
	if flat_idx is not None and max_abs_diff > 0.0:
		print(f"Worst abs mismatch at [{format_index(flat_idx, shape)}]: ref={worst_ref:.6e}, other={worst_other:.6e}")
	print_worst_pct_summary(a, b, finite_mask, shape, renorm_scales)


def verify_f8(file_a: str, file_b: str, shape: tuple[int, ...]) -> None:
	a_bits = load_f8_bits(file_a, shape)
	b_bits = load_f8_bits(file_b, shape)
	a = load_f8(file_a, shape)
	b = load_f8(file_b, shape)

	finite_mask = torch.isfinite(a) & torch.isfinite(b)
	diff = (a - b).abs()
	if finite_mask.any():
		finite_diff = diff[finite_mask]
		max_abs_diff = finite_diff.max().item()
		mean_abs_diff = finite_diff.mean().item()
		masked_diff = torch.where(finite_mask, diff, torch.full_like(diff, -1.0))
		flat_idx = int(masked_diff.reshape(-1).argmax().item())
		worst_ref = a.reshape(-1)[flat_idx].item()
		worst_other = b.reshape(-1)[flat_idx].item()
	else:
		max_abs_diff = math.nan
		mean_abs_diff = math.nan
		flat_idx = None
		worst_ref = math.nan
		worst_other = math.nan

	a_ord = f8_ordered_int(a_bits)
	b_ord = f8_ordered_int(b_bits)
	ulp_diff = (a_ord - b_ord).abs()
	same_value = (a == b) | (torch.isnan(a) & torch.isnan(b))
	bit_mismatch = a_bits != b_bits
	zero_sign_only = bit_mismatch & (a == 0) & (b == 0)
	raw_exact_match = int((a_bits == b_bits).sum().item())
	exact_match = int(same_value.sum().item())
	a_abs = a.abs()
	b_abs = b.abs()
	renorm_scales = variance_one_renorm_scales(a, b, finite_mask)
	pct_strs = max_pct_strs(a_abs, b_abs, finite_mask, renorm_scales)
	sign_flip_count, largest_sign_flip_mag = sign_flip_summary(a, b)
	total = a.numel()

	print("\n--- F8 E4M3FN Tensor Compare ---")
	print(f"A: {file_a}")
	print(f"B: {file_b}")
	print(f"Shape: [{', '.join(str(dim) for dim in shape)}]")
	print(f"Max abs diff:     {max_abs_diff:.6e}")
	print(f"Mean abs diff:    {mean_abs_diff:.6e}")
	print(f"Max pct diff:     {', '.join(pct_strs)}")
	print("  (MIN_MAG after scaling each tensor to variance 1:")
	print(f"               {PCT_BUCKET_LABELS})")
	print(f"Exact f8 value match: {exact_match}/{total} ({100.0 * exact_match / total:.2f}%)")
	print(f"Raw exact bytes:       {raw_exact_match}/{total} ({100.0 * raw_exact_match / total:.2f}%)")
	print(f"Signed-zero-only mismatches: {int(zero_sign_only.sum().item())}")
	print(f"Sign-flipped values: {sign_flip_count}, largest magnitude: {largest_sign_flip_mag:.6e}")
	print_ulp_summary(ulp_diff, total, a, b, shape)
	print(summarize_nonfinite("A non-finite", a))
	print(summarize_nonfinite("B non-finite", b))
	if flat_idx is not None and max_abs_diff > 0.0:
		worst_ref_bits = int(a_bits.reshape(-1)[flat_idx].item())
		worst_other_bits = int(b_bits.reshape(-1)[flat_idx].item())
		print(
			f"Worst abs mismatch at [{format_index(flat_idx, shape)}]: "
			f"ref={worst_ref:.6e} (0x{worst_ref_bits:02x}), "
			f"other={worst_other:.6e} (0x{worst_other_bits:02x})"
		)
	print_worst_pct_summary(a, b, finite_mask, shape, renorm_scales)


def verify(file_a: str, file_b: str, shape: tuple[int, ...], dtype: str) -> None:
	if dtype == "bf16":
		verify_bf16(file_a, file_b, shape)
	elif dtype == "f32":
		verify_f32(file_a, file_b, shape)
	elif dtype == "f8":
		verify_f8(file_a, file_b, shape)
	else:
		raise ValueError(f"Unsupported dtype: {dtype}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Compare two tensor .bin files")
	parser.add_argument("file_a", help="Reference tensor .bin file")
	parser.add_argument("file_b", help="Other tensor .bin file")
	parser.add_argument("--dtype", choices=["bf16", "f32", "f8"], default=None, help="Tensor element type")
	parser.add_argument("--shape", type=int, nargs="+", default=None, help="Optional tensor shape")
	args = parser.parse_args()

	dtype = args.dtype or infer_dtype_from_paths(args.file_a, args.file_b)
	bytes_per_elem = {"bf16": 2, "f32": 4, "f8": 1}[dtype]
	shape = infer_shape(args.file_a, args.file_b, args.shape, bytes_per_elem)
	verify(args.file_a, args.file_b, shape, dtype)


if __name__ == "__main__":
	main()
