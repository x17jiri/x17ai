import torch
from pathlib import Path
from jsonc import load_jsonc
import math

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "block.config.json"
TENSOR_DIR = ROOT / "tmp" / "block_torch"

my_device = torch.device("cpu") # or torch.device("cuda")
torch.set_default_device(my_device)
torch.set_default_dtype(torch.float32)

config = load_jsonc(CONFIG_PATH)
N_INPUTS = int(config["n_inputs"])
D_MODEL = int(config["d_model"])
N_HEADS = int(config["n_heads"])
HEAD_DIM = int(config["head_dim"])
ROPE_DIM = int(config["rope_dim"])
SPARSE_FAN_IN = int(config["qkv_fan_in"])
F_WIDTH = int(config["f_width"])
WINDOW_SIZE = int(config["window_size"])
L2_NORM_EPS = float(config["l2_norm_eps"])
ROPE_BASE = float(config["rope_base"])
QKVG_ROWS = 4 * N_HEADS * HEAD_DIM
ATTN_WIDTH = N_HEADS * HEAD_DIM
F_PROJ_OUTPUTS = 2 * F_WIDTH
SPARSE_SCALE = math.sqrt(D_MODEL / SPARSE_FAN_IN)
V_SCALE_FIX = 1.5

GELU_VAR_FIX_2 = 1.0 / (
	(1.0 / 3.0)
	+ (0.5 / math.pi) * (1.0 / math.sqrt(3.0))
)
GELU_VAR_FIX = math.sqrt(GELU_VAR_FIX_2)

# Each split projection should see unit total input variance, so each coordinate of the
# GeGLU output should contribute variance about 1 / branch_width.
ATTN_GEGLU_SCALE = math.sqrt(GELU_VAR_FIX_2 / ATTN_WIDTH)
F_GEGLU_SCALE = math.sqrt(GELU_VAR_FIX_2 / F_WIDTH)

def tensor_path(name: str) -> Path:
	return TENSOR_DIR / name

def load_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	path = tensor_path(tensor)
	raw = path.read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.int16)
	return data.view(torch.bfloat16).reshape(rows, cols).to(my_device).to(torch.float32)

def load_f32_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	path = tensor_path(tensor)
	raw = path.read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.float32)
	return data.reshape(rows, cols).to(my_device)

def variance_path(path: Path) -> Path:
	return path.with_name(f"{path.name}.var")

VARIANCE_WARNING_MIN_REL = 0.15
VARIANCE_WARNING_SIGMAS = 6.0
VARIANCE_WARNING_ZERO_ABS = 1e-6

def warn_if_variance_is_unexpected(path: Path, tensor: torch.Tensor, expected_variance: float | None) -> None:
	if expected_variance is None:
		return
	actual_variance = tensor.to(torch.float32).var(unbiased=False).item()
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
	path = tensor_path(file_name)
	path.parent.mkdir(parents=True, exist_ok=True)
	data = tensor.contiguous().to(torch.bfloat16).cpu()
	with path.open("wb") as output_file:
		output_file.write(data.view(torch.int16).numpy().tobytes())
	store_expected_variance(path, expected_variance)
	warn_if_variance_is_unexpected(path, data, expected_variance)
	shape_str = ", ".join(str(dim) for dim in data.shape)
	print(f"Created {path}: [{shape_str}] bfloat16")

def store_f32_tensor(tensor: torch.Tensor, file_name: str, expected_variance: float | None = None) -> None:
	path = tensor_path(file_name)
	path.parent.mkdir(parents=True, exist_ok=True)
	data = tensor.contiguous().cpu().to(torch.float32)
	with path.open("wb") as output_file:
		output_file.write(data.numpy().tobytes())
	store_expected_variance(path, expected_variance)
	warn_if_variance_is_unexpected(path, data, expected_variance)
	shape_str = ", ".join(str(dim) for dim in data.shape)
	print(f"Created {path}: [{shape_str}] float32")
