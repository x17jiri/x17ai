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
	raise KeyError(f"Missing config value, tried: {name}")

N_INPUTS = int(config_value("seq_len"))
MODEL_DIM = int(config_value("MODEL_DIM"))
N_HEADS = int(config_value("N_HEADS"))
HEAD_DIM = int(config_value("HEAD_DIM"))
F_WIDTH = int(config_value("F_WIDTH"))
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

TENSOR_FILE_EXTENSIONS = {".bin", ".safetensors"}
TENSOR_STORAGE_DTYPE_TAGS = {
	"f32": torch.float32,
	"i32": torch.int32,
	"bf16": torch.bfloat16,
	"i8": torch.int8,
	"f8": F8_DTYPE,
}

def tensor_file_extension(file_name: str) -> str:
	extension = Path(file_name).suffix
	if extension not in TENSOR_FILE_EXTENSIONS:
		supported = ", ".join(sorted(TENSOR_FILE_EXTENSIONS))
		raise ValueError(f"Unsupported tensor file extension {extension!r} in {file_name!r}; expected one of: {supported}")
	return extension

def tensor_storage_dtype_tag(file_name: str) -> str:
	extension = tensor_file_extension(file_name)
	stem = Path(file_name).stem
	base_name, separator, tag = stem.rpartition("_")
	if not separator or not base_name or tag not in TENSOR_STORAGE_DTYPE_TAGS:
		supported = ", ".join(f"_{tag}{extension}" for tag in TENSOR_STORAGE_DTYPE_TAGS)
		raise ValueError(f"Tensor file name must end with a dtype tag before the extension ({supported}): {file_name!r}")
	return tag

def tensor_storage_dtype(file_name: str) -> torch.dtype:
	tag = tensor_storage_dtype_tag(file_name)
	if tag == "f8":
		if F8_DTYPE is None:
			raise RuntimeError("This PyTorch build does not support float8_e4m3fn")
	dtype = TENSOR_STORAGE_DTYPE_TAGS[tag]
	if dtype is None:
		raise ValueError(f"Unsupported tensor storage dtype tag: {tag}")
	return dtype

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

def safetensors_tensor_name(file_name: str) -> str:
	stem = Path(file_name).stem
	base_name, _, _ = stem.rpartition("_")
	return base_name

def load_tensor_with_dtype(path: Path, rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
	raw = path.read_bytes()
	if dtype == torch.float32:
		data = torch.frombuffer(bytearray(raw), dtype=torch.float32)
		return load_stored_tensor(data, rows, cols, dtype)
	if dtype == torch.int8:
		data = torch.frombuffer(bytearray(raw), dtype=torch.int8)
		return load_stored_tensor(data, rows, cols, dtype)
	if dtype == torch.bfloat16:
		data = torch.frombuffer(bytearray(raw), dtype=torch.int16)
		return load_stored_tensor(data.view(torch.bfloat16), rows, cols, dtype)
	if F8_DTYPE is not None and dtype == F8_DTYPE:
		data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
		return load_stored_tensor(data.view(F8_DTYPE), rows, cols, dtype)
	raise ValueError(f"Unsupported tensor storage dtype: {dtype}")

def load_stored_tensor(data: torch.Tensor, rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
	if data.dtype != dtype:
		raise ValueError(f"Expected stored tensor dtype {tensor_storage_name(dtype)}, got {data.dtype}")
	if dtype == torch.int8:
		scaled = data.to(torch.float32) / 8.0
		data = torch.where(data == -128, torch.full_like(scaled, math.nan), scaled)
	return data.reshape(rows, cols).to(my_device).to(my_dtype)

def e4m3_ftz(tensor):
	# This number would be rounded to 2**-6, which is the smallest normal e4m3 value
	threshold = (1.0 + 15.0 / 16.0) * 2.0**-7

	# Flush sub-normals to zero
	tensor = torch.where(
		tensor.abs() < threshold,
		torch.zeros_like(tensor),
		tensor,
	)

	return tensor

def store_tensor_with_dtype(
	tensor: torch.Tensor,
	file_name: str,
	dtype: torch.dtype,
	expected_variance: float | None = None,
) -> None:
	expected_dtype = tensor_storage_dtype(file_name)
	if dtype != expected_dtype:
		raise ValueError(
			f"Tensor file name {file_name!r} declares {tensor_storage_name(expected_dtype)}, "
			f"but store call requested {tensor_storage_name(dtype)}"
		)
	path = tensor_path(file_name)
	path.parent.mkdir(parents=True, exist_ok=True)
	extension = tensor_file_extension(file_name)
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
		data = e4m3_ftz(data)
		stored = data.to(F8_DTYPE)
		raw = stored.view(torch.uint8).numpy().tobytes()
		warn_tensor = stored.to(torch.float32)
	else:
		raise ValueError(f"Unsupported tensor storage dtype: {dtype}")
	if extension == ".safetensors":
		from safetensors.torch import save_file

		save_file({safetensors_tensor_name(file_name): stored}, str(path))
	else:
		with path.open("wb") as output_file:
			output_file.write(raw)
	store_expected_variance(path, expected_variance)
	warn_if_variance_is_unexpected(path, warn_tensor, expected_variance)
	shape_str = ", ".join(str(dim) for dim in stored.shape)
	print(f"Created {path}: [{shape_str}] {tensor_storage_name(dtype)}")

def load_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	path = tensor_path(tensor)
	dtype = tensor_storage_dtype(tensor)
	if tensor_file_extension(tensor) == ".safetensors":
		from safetensors.torch import load_file

		tensors = load_file(str(path), device="cpu")
		if len(tensors) != 1:
			raise ValueError(f"Expected exactly one tensor in {path}, found {len(tensors)}")
		stored = next(iter(tensors.values())).contiguous()
		return load_stored_tensor(stored, rows, cols, dtype)
	return load_tensor_with_dtype(path, rows, cols, dtype)

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
