import torch
from pathlib import Path
from jsonc import load_jsonc
import math

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "block.config.json"
TENSOR_DIR = ROOT / "tmp" / "block_torch"

my_device = torch.device("cpu") # or torch.device("cuda")
torch.set_default_device(my_device)
my_dtype = torch.float64
torch.set_default_dtype(my_dtype)

config = load_jsonc(CONFIG_PATH)

def config_value(name: str):
	if name in config:
		return config[name]
	raise KeyError(f"Missing config value, tried: {', '.join(names)}")

N_INPUTS = int(config_value("seq_len"))
MODEL_DIM = int(config_value("MODEL_DIM"))
N_HEADS = int(config_value("N_HEADS"))
HEAD_DIM = int(config_value("HEAD_DIM"))
F_WIDTH = int(config_value("F_WIDTH"))
Y_SPARSE_FAN_IN = int(config_value("Y_SPARSE_FAN_IN"))
Y_SPARSE_STEP = int(config_value("Y_SPARSE_STEP"))
Y_SPARSE_BLOCK = int(config_value("Y_SPARSE_BLOCK"))
WINDOW_SIZE = int(config_value("WINDOW_SIZE"))
L2_NORM_EPS = float(config_value("L2_NORM_EPS"))

V_SCALE_FIX = float(config_value("V_SCALE_FIX"))

GELU_VAR_FIX_2 = 1.0 / (
	(1.0 / 3.0)
	+ (0.5 / math.pi) * (1.0 / math.sqrt(3.0))
)
GELU_VAR_FIX = math.sqrt(GELU_VAR_FIX_2)

# Each split projection should see unit total input variance, so each coordinate of the
# GeGLU output should contribute variance about 1 / branch_width.
ATTN_WIDTH = N_HEADS * HEAD_DIM
ATTN_GEGLU_SCALE = math.sqrt(GELU_VAR_FIX_2 / ATTN_WIDTH)
F_GEGLU_SCALE = math.sqrt(GELU_VAR_FIX_2 / F_WIDTH)

def tensor_path(name: str) -> Path:
	return TENSOR_DIR / name

F8_DTYPE = getattr(torch, "float8_e4m3fn", None)

def tensor_storage_dtype(file_name: str) -> torch.dtype:
	if file_name.endswith("_f32.bin"):
		return torch.float32
	if file_name.endswith("_i8.bin"):
		return torch.int8
	if file_name.endswith("_f8.bin"):
		if F8_DTYPE is None:
			raise RuntimeError("This PyTorch build does not support float8_e4m3fn")
		return F8_DTYPE
	return torch.bfloat16

def tensor_storage_name(dtype: torch.dtype) -> str:
	if dtype == torch.float32:
		return "float32"
	if dtype == torch.int32:
		return "int32"
	if dtype == torch.bfloat16:
		return "bfloat16"
	if dtype == torch.int8:
		return "int8"
	if F8_DTYPE is not None and dtype == F8_DTYPE:
		return "float8_e4m3fn"
	raise ValueError(f"Unsupported tensor storage dtype: {dtype}")

def load_tensor_with_dtype(path: Path, rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
	raw = path.read_bytes()
	if dtype == torch.float32:
		data = torch.frombuffer(bytearray(raw), dtype=torch.float32)
		return data.reshape(rows, cols).to(my_device).to(my_dtype)
	if dtype == torch.int8:
		data = torch.frombuffer(bytearray(raw), dtype=torch.int8)
		scaled = data.to(torch.float32) / 8.0
		data = torch.where(data == -128, torch.full_like(scaled, math.nan), scaled)
		return data.reshape(rows, cols).to(my_device).to(my_dtype)
	if dtype == torch.bfloat16:
		data = torch.frombuffer(bytearray(raw), dtype=torch.int16)
		return data.view(torch.bfloat16).reshape(rows, cols).to(my_device).to(my_dtype)
	if F8_DTYPE is not None and dtype == F8_DTYPE:
		data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
		return data.view(F8_DTYPE).reshape(rows, cols).to(my_device).to(my_dtype)
	raise ValueError(f"Unsupported tensor storage dtype: {dtype}")

def store_tensor_with_dtype(
	tensor: torch.Tensor,
	file_name: str,
	dtype: torch.dtype,
	expected_variance: float | None = None,
) -> None:
	path = tensor_path(file_name)
	path.parent.mkdir(parents=True, exist_ok=True)
	data = tensor.detach().contiguous().cpu()
	if dtype == torch.float32:
		stored = data.to(torch.float32)
		raw = stored.numpy().tobytes()
		warn_tensor = stored
	elif dtype == torch.int32:
		stored = data.to(torch.int32)
		raw = stored.numpy().tobytes()
		warn_tensor = stored.to(torch.float32)
	elif dtype == torch.int8:
		warn_tensor = data.to(data.to(torch.float32))
		stored = torch.clamp(torch.round(warn_tensor * 8.0), -127.0, +127.0).to(torch.int8)
		raw = stored.numpy().tobytes()
	elif dtype == torch.bfloat16:
		stored = data.to(torch.bfloat16)
		raw = stored.view(torch.int16).numpy().tobytes()
		warn_tensor = stored
	elif F8_DTYPE is not None and dtype == F8_DTYPE:
		stored = data.to(F8_DTYPE)
		raw = stored.view(torch.uint8).numpy().tobytes()
		warn_tensor = stored.to(torch.float32)
	else:
		raise ValueError(f"Unsupported tensor storage dtype: {dtype}")
	with path.open("wb") as output_file:
		output_file.write(raw)
	store_expected_variance(path, expected_variance)
	warn_if_variance_is_unexpected(path, warn_tensor, expected_variance)
	shape_str = ", ".join(str(dim) for dim in stored.shape)
	print(f"Created {path}: [{shape_str}] {tensor_storage_name(dtype)}")

def load_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	path = tensor_path(tensor)
	return load_tensor_with_dtype(path, rows, cols, tensor_storage_dtype(tensor))

def load_f32_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	path = tensor_path(tensor)
	return load_tensor_with_dtype(path, rows, cols, torch.float32)

def load_f8_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	if F8_DTYPE is None:
		raise RuntimeError("This PyTorch build does not support float8_e4m3fn")
	path = tensor_path(tensor)
	return load_tensor_with_dtype(path, rows, cols, F8_DTYPE)

def load_i8_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	path = tensor_path(tensor)
	return load_tensor_with_dtype(path, rows, cols, torch.int8)

def variance_path(path: Path) -> Path:
	return path.with_name(f"{path.name}.var")

VARIANCE_WARNING_MIN_REL = 0.15
VARIANCE_WARNING_SIGMAS = 6.0
VARIANCE_WARNING_ZERO_ABS = 1e-6

def warn_if_variance_is_unexpected(path: Path, tensor: torch.Tensor, expected_variance: float | None) -> None:
	if expected_variance is None:
		return
	actual_variance = tensor.to(torch.float64).var(unbiased=False).item()
	if expected_variance == 0.0:
		if abs(actual_variance) > VARIANCE_WARNING_ZERO_ABS:
			print(
				f"***** WARNING {path}: expected variance={expected_variance:.6e}, "
				f"actual variance={actual_variance:.6e}, "
				f"abs diff={abs(actual_variance):.6e}"
			)
		return
	numel = max(tensor.numel(), 2)
	rel_std = math.sqrt(2.0 / (numel - 1))
	rel_diff = abs(actual_variance - expected_variance) / min(actual_variance, expected_variance)
	rel_tol = max(VARIANCE_WARNING_MIN_REL, VARIANCE_WARNING_SIGMAS * rel_std)
	if rel_diff > rel_tol:
		print(
			f"***** WARNING {path}: expected variance={expected_variance:.6e}, "
			f"actual variance={actual_variance:.6e}, "
			f"rel diff={rel_diff:.2%}, tol={rel_tol:.2%}"
		)

def store_expected_variance(path: Path, expected_variance: float | None) -> None:
	var_path = variance_path(path)
	if expected_variance is None:
		if var_path.exists():
			var_path.unlink()
		return
	var_path.write_text(f"{expected_variance:.17g}\n", encoding="ascii")
	print(f"Created {var_path}: variance={expected_variance:.6e}")

def store_tensor(tensor: torch.Tensor, file_name: str, expected_variance: float | None = None) -> None:
	store_tensor_with_dtype(tensor, file_name, tensor_storage_dtype(file_name), expected_variance)

def store_f32_tensor(tensor: torch.Tensor, file_name: str, expected_variance: float | None = None) -> None:
	store_tensor_with_dtype(tensor, file_name, torch.float32, expected_variance)

def store_f8_tensor(tensor: torch.Tensor, file_name: str, expected_variance: float | None = None) -> None:
	if F8_DTYPE is None:
		raise RuntimeError("This PyTorch build does not support float8_e4m3fn")
	store_tensor_with_dtype(tensor, file_name, F8_DTYPE, expected_variance)

def store_i8_tensor(tensor: torch.Tensor, file_name: str, expected_variance: float | None = None) -> None:
	store_tensor_with_dtype(tensor, file_name, torch.int8, expected_variance)

def store_i32_tensor(tensor: torch.Tensor, file_name: str, expected_variance: float | None = None) -> None:
	store_tensor_with_dtype(tensor, file_name, torch.int32, expected_variance)
