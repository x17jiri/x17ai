#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

#include <memory>
#include <sstream>
#include <stdio.h>
#include <unordered_map>

#include <dlfcn.h>

#define FMT_HEADER_ONLY
#include <fmt/core.h>

#define X17_INLINE gnu::always_inline
#define X17_NO_INLINE gnu::noinline

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using usize = size_t;
using isize = std::common_type_t<
	std::make_signed_t<std::ptrdiff_t>,
	std::make_signed_t<std::size_t>>;

static std::atomic<bool> cuda_initialized = false;
static std::mutex cuda_init_mutex;

namespace x17ai {

	int cuda_init() {
		if (cuda_initialized.load(std::memory_order_acquire)) [[likely]] {
			return 0;
		}

		std::lock_guard<std::mutex> lock(cuda_init_mutex);
		if (cuda_initialized.load(std::memory_order_relaxed)) {
			return 0;
		}

		CUresult result = cuInit(0);
		if (result != CUDA_SUCCESS) {
			fmt::print(stderr, "Failed to initialize CUDA\n");
			return 1;
		}

		int cuda_device_count;
		cudaError_t error = cudaGetDeviceCount(&cuda_device_count);
		std::cout << "cudaGetDeviceCount(): " << cuda_device_count << std::endl;
		std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;

		/*void *handle = dlopen("libATen_cuda.so", RTLD_LAZY | RTLD_GLOBAL);
		if (!handle) {
			fprintf(stderr, "Failed to load libATen_cuda.so: %s\n", dlerror());
		}

		if (!torch::cuda::is_available()) {
			fmt::print(stderr, "TORCH_VERSION: {}\n", TORCH_VERSION);
			fmt::print(stderr, "CUDA is not available in Torch\n");
			fmt::print(
				stderr,
				"Torch CUDA device count: {}\n",
				torch::cuda::device_count()
			);
			return 1;
		}*/

		cuda_initialized.store(true, std::memory_order_release);
		return 0;
	}

	void *cuda_alloc_f32(i64 count) {
		assert(cuda_initialized);
		if (count <= 0) [[unlikely]] {
			fmt::print("Invalid count for CUDA allocation: {}\n", count);
			return nullptr;
		}

		torch::Device device(torch::kCUDA, 0);

		torch::TensorOptions options =
			torch::TensorOptions().device(device).dtype(torch::kFloat32);

		auto tensor =
			std::make_unique<torch::Tensor>(torch::empty({count}, options));

		return tensor.release();
	}

	void cuda_free(void *ptr) {
		auto tensor =
			std::unique_ptr<torch::Tensor>(static_cast<torch::Tensor *>(ptr));
	}

	/*
	fn x17ai_cuda_new_kernel(source
							 : *const std::ffi::c_char, len
							 : usize, ) -> *const std::ffi::c_void;
	fn x17ai_cuda_del_kernel(kernel : *const std::ffi::c_void);

	fn x17ai_cuda_run_kernel(kernel
							 : *const std::ffi::c_void, o
							 : *const KernelOutput, elem_args
							 : *const KernelElemArg, reduce_args
							 : *const KernelReduceArg, const_args
							 : *const f64, ) -> std::ffi::c_int;
	*/
} // namespace x17ai

extern "C" {
	int x17ai_cuda_init() {
		try {
			return x17ai::cuda_init();
		} catch (...) {
			fmt::print(stderr, "CUDA initialization threw an exception\n");
			return 1;
		}
	}

	void *x17ai_cuda_alloc_f32(i64 count) {
		try {
			return x17ai::cuda_alloc_f32(count);
		} catch (...) {
			fmt::print(stderr, "CUDA allocation threw an exception\n");
			return nullptr;
		}
	}

	void x17ai_cuda_free(void *ptr) {
		try {
			x17ai::cuda_free(ptr);
		} catch (...) {
			fmt::print(stderr, "CUDA free threw an exception\n");
		}
	}

	/*
	fn x17ai_cuda_new_kernel(source
							 : *const std::ffi::c_char, len
							 : usize, ) -> *const std::ffi::c_void;
	fn x17ai_cuda_del_kernel(kernel : *const std::ffi::c_void);

	fn x17ai_cuda_run_kernel(kernel
							 : *const std::ffi::c_void, o
							 : *const KernelOutput, elem_args
							 : *const KernelElemArg, reduce_args
							 : *const KernelReduceArg, const_args
							 : *const f64, ) -> std::ffi::c_int;
	*/
}

/*
// Cache for compiled modules
static std::unordered_map<std::string, CUmodule> module_cache;

// Compile CUDA source code at runtime
extern "C" bool
compile_cuda_kernel(const char *source_code, const char *kernel_name) {
	if (!x17ai::cuda_init())
		return false;

	// Check if already compiled
	std::string key(kernel_name);
	if (module_cache.find(key) != module_cache.end()) {
		return true; // Already compiled
	}

	// Create NVRTC program
	nvrtcProgram prog;
	nvrtcResult result = nvrtcCreateProgram(
		&prog,
		source_code,
		"kernel.cu",
		0,
		nullptr,
		nullptr
	);
	if (result != NVRTC_SUCCESS) {
		fmt::print(
			stderr,
			"Failed to create NVRTC program: {}\n",
			nvrtcGetErrorString(result)
		);
		return false;
	}

	// Compile options
	const char *opts[] = {
		"--gpu-architecture=compute_75", // Adjust for your GPU
		"--use_fast_math",
		"--std=c++17"
	};

	// Compile the program
	result = nvrtcCompileProgram(prog, 3, opts);
	if (result != NVRTC_SUCCESS) {
		// Get compilation log
		usize log_size;
		nvrtcGetProgramLogSize(prog, &log_size);
		std::string log(log_size, '\0');
		nvrtcGetProgramLog(prog, &log[0]);
		fmt::print(stderr, "Compilation failed:\n{}\n", log);
		nvrtcDestroyProgram(&prog);
		return false;
	}

	// Get PTX
	usize ptx_size;
	nvrtcGetPTXSize(prog, &ptx_size);
	std::string ptx(ptx_size, '\0');
	nvrtcGetPTX(prog, &ptx[0]);

	// Load module
	CUmodule module;
	CUresult cu_result =
		cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr);
	if (cu_result != CUDA_SUCCESS) {
		fmt::print(stderr, "Failed to load CUDA module\n");
		nvrtcDestroyProgram(&prog);
		return false;
	}

	// Cache the module
	module_cache[key] = module;

	// Cleanup
	nvrtcDestroyProgram(&prog);
	return true;
}

// Execute a compiled kernel
extern "C" bool execute_kernel(
	const char *kernel_name,
	const char *function_name,
	void **args,
	int num_args,
	int grid_x,
	int grid_y,
	int grid_z,
	int block_x,
	int block_y,
	int block_z
) {
	if (!cuda_initialized)
		return false;

	std::string key(kernel_name);
	auto it = module_cache.find(key);
	if (it == module_cache.end()) {
		fmt::print(stderr, "Kernel not found: {}\n", kernel_name);
		return false;
	}

	CUmodule module = it->second;
	CUfunction function;

	CUresult result = cuModuleGetFunction(&function, module, function_name);
	if (result != CUDA_SUCCESS) {
		fmt::print(stderr, "Failed to get function: {}\n", function_name);
		return false;
	}

	// Launch kernel
	result = cuLaunchKernel(
		function,
		grid_x,
		grid_y,
		grid_z, // grid dimensions
		block_x,
		block_y,
		block_z, // block dimensions
		0, // shared memory
		0, // stream
		args, // arguments
		nullptr // extra
	);

	if (result != CUDA_SUCCESS) {
		fmt::print(stderr, "Failed to launch kernel\n");
		return false;
	}

	// Synchronize
	cuCtxSynchronize();
	return true;
}

// Helper function for tensor multiplication specifically
extern "C" bool tensor_multiply_dynamic(
	const float *a,
	const float *b,
	float *result,
	int size,
	const char *kernel_source
) {
	// Compile kernel if not already compiled
	if (!compile_cuda_kernel(kernel_source, "tensor_multiply")) {
		return false;
	}

	// Prepare arguments
	void *args[] = {(void *)&a, (void *)&b, (void *)&result, (void *)&size};

	// Calculate grid dimensions
	int block_size = 256;
	int grid_size = (size + block_size - 1) / block_size;

	// Execute kernel
	return execute_kernel(
		"tensor_multiply", // kernel name (cache key)
		"tensor_multiply_kernel", // function name in CUDA code
		args,
		4, // arguments
		grid_size,
		1,
		1, // grid dimensions
		block_size,
		1,
		1 // block dimensions
	);
}

// Existing libtorch tensor wrapper (simplified)
extern "C" bool multiply_tensors_with_dynamic_kernel(
	void *tensor_a_ptr,
	void *tensor_b_ptr,
	void *result_ptr,
	const char *kernel_source
) {
	auto *tensor_a = static_cast<torch::Tensor *>(tensor_a_ptr);
	auto *tensor_b = static_cast<torch::Tensor *>(tensor_b_ptr);
	auto *result = static_cast<torch::Tensor *>(result_ptr);

	// Ensure tensors are on CUDA and contiguous
	auto a_cuda = tensor_a->to(torch::kCUDA).contiguous();
	auto b_cuda = tensor_b->to(torch::kCUDA).contiguous();
	*result = torch::empty_like(a_cuda);

	// Get raw pointers
	const float *a_ptr = a_cuda.data_ptr<float>();
	const float *b_ptr = b_cuda.data_ptr<float>();
	float *result_ptr_raw = result->data_ptr<float>();
	int size = a_cuda.numel();

	return tensor_multiply_dynamic(
		a_ptr,
		b_ptr,
		result_ptr_raw,
		size,
		kernel_source
	);
}
*/
