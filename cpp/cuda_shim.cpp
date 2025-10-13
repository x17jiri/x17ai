#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <atomic>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <unordered_map>
#include <cassert>
#include <span>

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
	std::make_signed_t<std::size_t>
>;

using DevicePtr = u64;

static_assert(std::is_unsigned_v<CUdeviceptr>);
static_assert(sizeof(DevicePtr) >= sizeof(CUdeviceptr));

static std::atomic<bool> cuda_initialized = false;
static std::mutex cuda_init_mutex;

struct StaticString {
	const char *data;
	usize len;

	consteval StaticString(char const *str):
		data(str),
		len(strlen(str))
	{}
};

struct Err {
	StaticString const *message;

	inline Err(StaticString const *m):
		message(m)
	{}
};

// The decision of whether the result is ok or not
// is based on `result == null`, not on `error == null`.
// That way:
// - we can avoid reading `error` most of the time.
// - we can safely convert result to NonNull in Rust.
// - we can use something like `Ok(result, "Err: Cuda returned null")` in the C++ code,
// avoiding one condition.
struct PtrResult {
	void *result;
	StaticString const *error;

	inline PtrResult(void *ptr, StaticString const *err):
		result(ptr),
		error(err)
	{
		assert(err != nullptr);
	}

	inline PtrResult(Err err):
		result(nullptr),
		error(err.message)
	{}

	inline bool is_ok() const {
		return result != nullptr;
	}

	inline bool is_err() const {
		return result == nullptr;
	}
};

// `err` will be used if `ptr == nullptr`
inline PtrResult Ok(void *ptr, StaticString const *err) {
	return PtrResult(ptr, err);
}

struct DevicePtrResult {
	DevicePtr result;
	StaticString const *error;

	inline DevicePtrResult(DevicePtr ptr):
		result(ptr),
		error(nullptr)
	{}

	inline DevicePtrResult(Err err):
		result(0),
		error(err.message)
	{}

	inline bool is_ok() const {
		return error == nullptr;
	}

	inline bool is_err() const {
		return error != nullptr;
	}
};

inline DevicePtrResult Ok(DevicePtr ptr) {
	return DevicePtrResult(ptr);
}

struct VoidResult {
	StaticString const *error;

	inline VoidResult(StaticString const *err):
		error(err)
	{}

	inline VoidResult(Err err):
		error(err.message)
	{}

	inline bool is_ok() const {
		return error == nullptr;
	}

	inline bool is_err() const {
		return error != nullptr;
	}
};

inline VoidResult Ok() {
	return VoidResult(nullptr);
}

struct FfiSpan {
	u8 *ptr;
	usize len;
};

struct FfiBufferVMT {
	FfiSpan (*span)(void *self);
	FfiSpan (*buf_span)(void *self);
	FfiSpan (*extend)(void *self, usize additional);
};

struct FfiBuffer {
	void *instance;
	FfiBufferVMT const *vmt;

	inline std::span<char> span() {
		FfiSpan s = vmt->span(instance);
		return std::span(reinterpret_cast<char *>(s.ptr), s.len);
	}

	inline std::span<char> buf_span() {
		FfiSpan s = vmt->buf_span(instance);
		return std::span(reinterpret_cast<char *>(s.ptr), s.len);
	}

	inline std::span<char> extend(usize additional) {
		FfiSpan s = vmt->extend(instance, additional);
		return std::span(reinterpret_cast<char *>(s.ptr), s.len);
	}

	bool write(std::string_view str) {
		size_t len = str.size();
		std::span buf = extend(len);
		if (buf.size() != len) [[unlikely]] {
			return false;
		}
		std::copy(str.begin(), str.end(), buf.data());
		return true;
	}

	bool writeln(std::string_view str) {
		size_t len = str.size();
		std::span buf = extend(len + 1);
		if (buf.size() != len) [[unlikely]] {
			return false;
		}
		std::copy(str.begin(), str.end(), buf.data());
		buf[len] = '\n';
		return true;
	}
};

extern "C" {
	std::string_view cuErrStr(CUresult result) {
		const char *err_str = nullptr;
		if (cuGetErrorString(result, &err_str) != CUDA_SUCCESS) [[unlikely]] {
			return "Unknown CUDA error";
		}
		return err_str;
	}

	void x17ai_test() {
		// init
		CUresult result = cuInit(0);
		fmt::print("cuInit result: {}\n", cuErrStr(result));

		// retain
		CUcontext ctx;
		result = cuDevicePrimaryCtxRetain(&ctx, 0);
		fmt::print("cuDevicePrimaryCtxRetain result: {}\n", cuErrStr(result));

		// getctx
		CUcontext current_ctx;
		result = cuCtxGetCurrent(&current_ctx);
		fmt::print("cuCtxGetCurrent result: {}\n", cuErrStr(result));
		fmt::print("cuCtxGetCurrent ctx: {}\n", (void *)current_ctx);

		// setctx
		result = cuCtxSetCurrent(ctx);
		fmt::print("cuCtxSetCurrent result: {}\n", cuErrStr(result));

		// create stream
		cudaStream_t stream;
		result = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
		fmt::print("cuStreamCreate result: {}\n", cuErrStr(result));

		// reset ctx
		result = cuCtxSetCurrent(nullptr);
		fmt::print("cuCtxSetCurrent(nullptr) result: {}\n", cuErrStr(result));

		// create ctx
		CUcontext new_ctx;
		result = cuCtxCreate(&new_ctx, CU_CTX_SCHED_AUTO, 0);
		fmt::print("cuCtxCreate result: {}\n", cuErrStr(result));

		// get ctx
		result = cuCtxGetCurrent(&current_ctx);
		fmt::print("cuCtxGetCurrent result: {}\n", cuErrStr(result));
		fmt::print("cuCtxGetCurrent ctx: {}\n", (void *)current_ctx);

		// malloc
		CUdeviceptr d_ptr;
		result = cuMemAllocAsync(&d_ptr, 1024, stream);
		fmt::print("cuMemAllocAsync result: {}\n", cuErrStr(result));

		// memcpy
		u8 h_data[1024] = {7};
		result = cuMemcpyHtoDAsync(d_ptr, h_data, sizeof(h_data), stream);
		fmt::print("cuMemcpyHtoDAsync result: {}\n", cuErrStr(result));

		// release
		result = cuDevicePrimaryCtxRelease(0);
		fmt::print("cuDevicePrimaryCtxRelease result: {}\n", cuErrStr(result));
	}

	auto x17ai_cuda_open_context(usize device_id) -> PtrResult {
		try {
			CUresult result;
			if (!cuda_initialized.load(std::memory_order_acquire)) [[unlikely]] {
				std::lock_guard<std::mutex> lock(cuda_init_mutex);
				if (!cuda_initialized.load(std::memory_order_relaxed)) {
					result = cuInit(0);
					if (result != CUDA_SUCCESS) [[unlikely]] {
						static StaticString const message =
							"x17ai_cuda_open_context(): Failed to initialize CUDA";
						return Err(&message);
					}
					cuda_initialized.store(true, std::memory_order_release);
				}
			}

			using MyUnsignedId = decltype(device_id);
			static_assert(std::is_unsigned_v<MyUnsignedId>);
			using CuUnsignedId = std::make_unsigned_t<CUdevice>;
			using CommonId = std::common_type_t<MyUnsignedId, CuUnsignedId>;
			constexpr CommonId MAX_ID = CuUnsignedId(std::numeric_limits<CUdevice>::max());

			if (CommonId(device_id) > MAX_ID) [[unlikely]] {
				static StaticString const message =
					"x17ai_cuda_open_context(): CUDA device ID out of range";
				return Err(&message);
			}

			CUcontext ctx;
			result = cuDevicePrimaryCtxRetain(&ctx, device_id);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				static StaticString const message =
					"x17ai_cuda_open_context(): Failed to retain primary CUDA context";
				return Err(&message);
			}

			static StaticString const message =
				"x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() returned nullptr";
			return Ok(ctx, &message);
		} catch (...) {
			static StaticString const message = "x17ai_cuda_close_context(): exception thrown";
			return Err(&message);
		}
	}

	auto x17ai_cuda_close_context(usize device_id) -> VoidResult {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));

			using MyUnsignedId = decltype(device_id);
			static_assert(std::is_unsigned_v<MyUnsignedId>);
			using CuUnsignedId = std::make_unsigned_t<CUdevice>;
			using CommonId = std::common_type_t<MyUnsignedId, CuUnsignedId>;
			constexpr CommonId MAX_ID = CuUnsignedId(std::numeric_limits<CUdevice>::max());

			if (CommonId(device_id) > MAX_ID) [[unlikely]] {
				static StaticString const message =
					"x17ai_cuda_close_context(): CUDA device ID out of range";
				return Err(&message);
			}

			CUresult result = cuDevicePrimaryCtxRelease(device_id);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				static StaticString const message =
					"x17ai_cuda_close_context(): Failed to release primary CUDA context";
				return Err(&message);
			}
			return Ok();
		} catch (...) {
			static StaticString const message = "x17ai_cuda_close_context(): exception thrown";
			return Err(&message);
		}
	}

	auto x17ai_cuda_open_stream(void *context) -> PtrResult {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(context != nullptr);
			CUcontext ctx = static_cast<CUcontext>(context);

			CUresult result = cuCtxPushCurrent(ctx);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				static StaticString const message =
					"x17ai_cuda_open_stream(): Failed to push CUDA context";
				return Err(&message);
			}

			CUstream cu_stream;
			result = cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				static StaticString const message =
					"x17ai_cuda_open_stream(): Failed to create CUDA stream";
				return Err(&message);
			}

			cuCtxPopCurrent(&ctx);

			static StaticString const message =
				"x17ai_cuda_open_stream(): cuStreamCreate() returned nullptr";
			return Ok(cu_stream, &message);
		} catch (...) {
			static StaticString const message = "x17ai_cuda_open_stream(): exception thrown";
			return Err(&message);
		}
	}

	auto x17ai_cuda_close_stream(void *stream) -> VoidResult {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = static_cast<CUstream>(stream);

			CUresult result = cuStreamDestroy(cu_stream);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				static StaticString const message =
					"x17ai_cuda_close_stream(): Failed to destroy CUDA stream";
				return Err(&message);
			}
			return Ok();
		} catch (...) {
			static StaticString const message = "x17ai_cuda_close_stream(): exception thrown";
			return Err(&message);
		}
	}

	auto x17ai_cuda_alloc(void *stream, usize bytes) -> DevicePtrResult {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = static_cast<CUstream>(stream);

			CUdeviceptr memory = 0;
			CUresult result = cuMemAllocAsync(&memory, bytes, cu_stream);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				static StaticString const message = "x17ai_cuda_alloc(): cuMemAllocAsync() failed";
				return Err(&message);
			}
			return Ok(memory);
		} catch (...) {
			static StaticString const message = "x17ai_cuda_alloc(): exception thrown";
			return Err(&message);
		}
	}

	auto x17ai_cuda_free(void *stream, DevicePtr ptr) -> VoidResult {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = static_cast<CUstream>(stream);

			CUresult result = cuMemFreeAsync(static_cast<CUdeviceptr>(ptr), cu_stream);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				static StaticString const message = "x17ai_cuda_free(): cuMemFreeAsync() failed";
				return Err(&message);
			}
			return Ok();
		} catch (...) {
			static StaticString const message = "x17ai_cuda_free(): exception thrown";
			return Err(&message);
		}
	}

	auto x17ai_cuda_upload_data(
		void *stream,
		const u8 *src,
		DevicePtr dst,
		usize offset_bytes,
		usize count_bytes
	) -> VoidResult {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			assert(src != nullptr);
			CUstream cu_stream = static_cast<CUstream>(stream);

			CUresult result = cuMemcpyHtoDAsync(
				static_cast<CUdeviceptr>(dst) + static_cast<CUdeviceptr>(offset_bytes),
				src,
				count_bytes,
				cu_stream
			);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				static StaticString const message =
					"x17ai_cuda_upload_data(): cuMemcpyHtoDAsync() failed";
				return Err(&message);
			}
			return Ok();
		} catch (...) {
			static StaticString const message = "x17ai_cuda_upload_data(): exception thrown";
			return Err(&message);
		}
	}

	auto x17ai_cuda_download_data(
		void *stream,
		DevicePtr src,
		u8 *dst,
		usize offset_bytes,
		usize count_bytes
	) -> VoidResult {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			assert(dst != nullptr);
			CUstream cu_stream = static_cast<CUstream>(stream);

			CUresult result = cuMemcpyDtoHAsync(
				dst,
				static_cast<CUdeviceptr>(src) + static_cast<CUdeviceptr>(offset_bytes),
				count_bytes,
				cu_stream
			);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				static StaticString const message =
					"x17ai_cuda_download_data(): cuMemcpyDtoHAsync() failed";
				return Err(&message);
			}
			return Ok();
		} catch (...) {
			static StaticString const message = "x17ai_cuda_download_data(): exception thrown";
			return Err(&message);
		}
	}

	auto x17ai_cuda_compile_kernel(void *stream, char const *source, FfiBuffer ptx, FfiBuffer log)
		-> VoidResult {
		assert(stream != nullptr);
		assert(cuda_initialized.load(std::memory_order_acquire));

		// Create NVRTC program
		nvrtcProgram prog;
		nvrtcResult result = nvrtcCreateProgram(&prog, source, "kernel.cu", 0, nullptr, nullptr);
		if (result != NVRTC_SUCCESS) {
			log.write("nvrtcCreateProgram() failed with error: ");
			log.writeln(nvrtcGetErrorString(result));
			return 1;
		}

		// Compile the program
		auto options = std::to_array<const char *>(
			{"--gpu-architecture=compute_75", // Adjust for your GPU
			 "--use_fast_math",
			 "--std=c++17"}
		);
		result = nvrtcCompileProgram(prog, options.size(), options.data());
		if (result != NVRTC_SUCCESS) {
			// Get compilation log
			usize log_size;
			result = nvrtcGetProgramLogSize(prog, &log_size);
			if (result != NVRTC_SUCCESS) {
				// TODO
			}
			auto log = buffer.reserve_exact(log_size);
			if (buffer.set_len(log_size)) [[likely]] {
				result = nvrtcGetProgramLog(prog, log.data());
				if (result != NVRTC_SUCCESS) {
					// TODO
				}
			}
			result = nvrtcDestroyProgram(&prog);
			if (result != NVRTC_SUCCESS) {
				// TODO
			}
			return 1;
		}

		// Get PTX
		usize ptx_size;
		result = nvrtcGetPTXSize(prog, &ptx_size);
		if (result != NVRTC_SUCCESS) {
			// TODO
		}
		auto ptx = buffer.reserve_exact(ptx_size);
		if (!buffer.set_len(ptx_size)) [[unlikely]] {
			// TODO - try to log error
			return 1;
		}
		result = nvrtcGetPTX(prog, ptx.data());
		if (result != NVRTC_SUCCESS) {
			// TODO
		}
		result = nvrtcDestroyProgram(&prog);
		if (result != NVRTC_SUCCESS) {
			// TODO
		}
		return 0;
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
