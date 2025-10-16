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

using f32 = float;
using f64 = double;

using usize = size_t;
using isize = std::common_type_t<
	std::make_signed_t<std::ptrdiff_t>,
	std::make_signed_t<std::size_t>
>;

static std::atomic<bool> cuda_initialized = false;
static std::mutex cuda_init_mutex;

struct FfiSpan {
	u8 *ptr;
	usize len;
};

struct FfiBufferVMT {
	FfiSpan (*span)(void *self) noexcept;
	FfiSpan (*buf_span)(void *self) noexcept;
	FfiSpan (*extend)(void *self, usize additional) noexcept;
	void (*clear)(void *self) noexcept;
};

struct FfiBuffer {
	void *instance;
	FfiBufferVMT const *vmt;

	inline std::span<char> span() noexcept {
		FfiSpan s = vmt->span(instance);
		return std::span(reinterpret_cast<char *>(s.ptr), s.len);
	}

	inline std::span<char> buf_span() noexcept {
		FfiSpan s = vmt->buf_span(instance);
		return std::span(reinterpret_cast<char *>(s.ptr), s.len);
	}

	inline std::span<char> extend(usize additional) noexcept {
		FfiSpan s = vmt->extend(instance, additional);
		return std::span(reinterpret_cast<char *>(s.ptr), s.len);
	}

	bool write() noexcept {
		return true;
	}

	bool write(std::string_view str) noexcept {
		size_t len = str.size();
		std::span buf = extend(len);
		if (buf.size() != len) [[unlikely]] {
			return false;
		}
		std::copy(str.begin(), str.end(), buf.data());
		return true;
	}

	bool write(int value) noexcept {
		return write(fmt::to_string(value));
	}

	bool write(CUresult e) noexcept {
		const char *err_str = nullptr;
		if (cuGetErrorString(e, &err_str) == CUDA_SUCCESS) [[likely]] {
			return write(err_str);
		} else {
			return write("Unknown CUDA error code ", int(e));
		}
	}

	bool write(nvrtcResult e) noexcept {
		return write(nvrtcGetErrorString(e));
	}

	template<typename A, typename B, typename... CS>
	bool write(A const &a, B const &b, CS const &... cs) noexcept {
		if (!write(a)) {
			return false;
		}
		return write(b, cs...);
	}

	void clear() noexcept {
		vmt->clear(instance);
	}
};

// #[repr(C)]
// pub struct KernelOutput {
// 	pub size: [usize; 2],
// 	pub stride_bytes: [usize; 2],
// 	pub buf: DevicePtr,
// 	pub offset_bytes: usize,
// 	pub reduction_size: usize,
// }

struct KernelOutput {
	usize size[2];
	usize stride_bytes[2];
	void *buf;
	usize offset_bytes;
	usize reduction_size;
};

// #[repr(C)]
// pub struct KernelElemArg {
// 	pub stride_bytes: [usize; 2],
// 	pub buf: DevicePtr,
// 	pub offset_bytes: usize,
// }

struct KernelElemArg {
	usize stride_bytes[2];
	void *buf;
	usize offset_bytes;
};

// #[repr(C)]
// pub struct KernelReduceArg {
// 	pub stride_bytes: [usize; 3],
// 	pub buf: DevicePtr,
// 	pub offset_bytes: usize,
// }

struct KernelReduceArg {
	usize stride_bytes[3];
	void *buf;
	usize offset_bytes;
};



struct CudaContextHandle;
struct CudaStreamHandle;
struct CudaKernelHandle;
struct CudaDeviceData;

inline CUdeviceptr to_dev_ptr(CudaDeviceData *ptr) noexcept {
	using U = std::make_unsigned_t<CUdeviceptr>;
	return CUdeviceptr(U(reinterpret_cast<uintptr_t>(ptr)));
}

inline CudaDeviceData *from_dev_ptr(CUdeviceptr ptr) noexcept {
	using U = std::make_unsigned_t<CUdeviceptr>;
	static_assert(sizeof(uintptr_t) >= sizeof(CUdeviceptr));
	static_assert(sizeof(CudaDeviceData *) >= sizeof(CUdeviceptr));
	return reinterpret_cast<CudaDeviceData *>(uintptr_t(U(ptr)));
}

extern "C" {
	/*void x17ai_test() {
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
	}*/

	CudaContextHandle *x17ai_cuda_open_context(usize device_id, FfiBuffer err) noexcept {
		try {
			CUresult e;
			if (!cuda_initialized.load(std::memory_order_acquire)) [[unlikely]] {
				std::lock_guard<std::mutex> lock(cuda_init_mutex);
				if (!cuda_initialized.load(std::memory_order_relaxed)) {
					e = cuInit(0);
					if (e != CUDA_SUCCESS) [[unlikely]] {
						err.write("x17ai_cuda_open_context(): cuInit() failed: ", e);
						return nullptr;
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
				err.write("x17ai_cuda_open_context(): CUDA device ID out of range");
				return nullptr;
			}

			CUcontext ctx = nullptr;
			e = cuDevicePrimaryCtxRetain(&ctx, device_id);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() failed: ", e);
				return nullptr;
			}
			if (ctx == nullptr) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() returned nullptr");
				return nullptr;
			}

			return reinterpret_cast<CudaContextHandle *>(ctx);
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_open_context(): exception thrown: ", e.what());
			return nullptr;
		} catch (...) {
			err.write("x17ai_cuda_open_context(): unknown exception thrown");
			return nullptr;
		}
	}

	int x17ai_cuda_close_context(usize device_id, FfiBuffer err) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			CUresult e;

			using MyUnsignedId = decltype(device_id);
			static_assert(std::is_unsigned_v<MyUnsignedId>);
			using CuUnsignedId = std::make_unsigned_t<CUdevice>;
			using CommonId = std::common_type_t<MyUnsignedId, CuUnsignedId>;
			constexpr CommonId MAX_ID = CuUnsignedId(std::numeric_limits<CUdevice>::max());

			if (CommonId(device_id) > MAX_ID) [[unlikely]] {
				err.write("x17ai_cuda_close_context(): CUDA device ID out of range");
				return -1;
			}

			e = cuDevicePrimaryCtxRelease(device_id);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_close_context(): cuDevicePrimaryCtxRelease() failed: ", e);
				return -1;
			}
			return 0;
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_close_context(): exception thrown: ", e.what());
			return -1;
		} catch (...) {
			err.write("x17ai_cuda_close_context(): unknown exception thrown");
			return -1;
		}
	}

	CudaStreamHandle *x17ai_cuda_open_stream(CudaContextHandle *context, FfiBuffer err) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(context != nullptr);
			CUcontext ctx = reinterpret_cast<CUcontext>(context);
			CUresult e;

			e = cuCtxPushCurrent(ctx);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_open_stream(): cuCtxPushCurrent() failed: ", e);
				return nullptr;
			}

			CUstream cu_stream;
			e = cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING);

			[[maybe_unused]] auto _e = cuCtxPopCurrent(&ctx);

			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_open_stream(): cuStreamCreate() failed: ", e);
				return nullptr;
			}
			if (cu_stream == nullptr) [[unlikely]] {
				err.write("x17ai_cuda_open_stream(): cuStreamCreate() returned nullptr");
				return nullptr;
			}

			return reinterpret_cast<CudaStreamHandle *>(cu_stream);
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_open_stream(): exception thrown: ", e.what());
			return nullptr;
		} catch (...) {
			err.write("x17ai_cuda_open_stream(): unknown exception thrown");
			return nullptr;
		}
	}

	int x17ai_cuda_close_stream(CudaStreamHandle *stream, FfiBuffer err) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuStreamDestroy(cu_stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_close_stream(): cuStreamDestroy() failed: ", e);
				return -1;
			}
			return 0;
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_close_stream(): exception thrown: ", e.what());
			return -1;
		} catch (...) {
			err.write("x17ai_cuda_close_stream(): unknown exception thrown");
			return -1;
		}
	}

	CudaDeviceData *x17ai_cuda_alloc(CudaStreamHandle *stream, usize bytes, FfiBuffer err) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			CUdeviceptr memory = 0;
			e = cuMemAllocAsync(&memory, bytes, cu_stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_alloc(): cuMemAllocAsync() failed: ", e);
				return nullptr;
			}
			if (memory == 0) [[unlikely]] {
				err.write("x17ai_cuda_alloc(): cuMemAllocAsync() returned null pointer");
			}
			return from_dev_ptr(memory);
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_alloc(): exception thrown: ", e.what());
			return nullptr;
		} catch (...) {
			err.write("x17ai_cuda_alloc(): unknown exception thrown");
			return nullptr;
		}
	}

	int x17ai_cuda_free(CudaStreamHandle *stream, CudaDeviceData *ptr, FfiBuffer err) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuMemFreeAsync(to_dev_ptr(ptr), cu_stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_free(): cuMemFreeAsync() failed: ", e);
				return -1;
			}
			return 0;
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_free(): exception thrown: ", e.what());
			return -1;
		} catch (...) {
			err.write("x17ai_cuda_free(): unknown exception thrown");
			return -1;
		}
	}

	int x17ai_cuda_upload_data(
		CudaStreamHandle *stream,
		u8 const *src,
		CudaDeviceData *dst,
		usize offset_bytes,
		usize size_bytes,
		FfiBuffer err
	) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			assert(src != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuMemcpyHtoDAsync(
				to_dev_ptr(dst) + offset_bytes,
				src,
				size_bytes,
				cu_stream
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_upload_data(): cuMemcpyHtoDAsync() failed: ", e);
				return -1;
			}
			return 0;
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_upload_data(): exception thrown: ", e.what());
			return -1;
		} catch (...) {
			err.write("x17ai_cuda_upload_data(): unknown exception thrown");
			return -1;
		}
	}

	int x17ai_cuda_download_data(
		CudaStreamHandle *stream,
		CudaDeviceData *src,
		u8 *dst,
		usize offset_bytes,
		usize size_bytes,
		FfiBuffer err
	) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			assert(dst != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuMemcpyDtoHAsync(
				dst,
				to_dev_ptr(src) + offset_bytes,
				size_bytes,
				cu_stream
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_download_data(): cuMemcpyDtoHAsync() failed: ", e);
				return -1;
			}
			return 0;
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_download_data(): exception thrown: ", e.what());
			return -1;
		} catch (...) {
			err.write("x17ai_cuda_download_data(): unknown exception thrown");
			return -1;
		}
	}

	void x17ai_cuda_compile_kernel(
		CudaStreamHandle *stream,
		char const *source,
		FfiBuffer ptx,
		FfiBuffer log
	) noexcept {
		assert(stream != nullptr);
		assert(cuda_initialized.load(std::memory_order_acquire));
		nvrtcResult e;

		// Create NVRTC program
		nvrtcProgram prog;
		e = nvrtcCreateProgram(&prog, source, "kernel.cu", 0, nullptr, nullptr);
		if (e != NVRTC_SUCCESS) [[unlikely]] {
			log.write("x17ai_cuda_compile_kernel(): nvrtcCreateProgram() failed: ", e);
			return;
		}

		// Compile the program
		auto options = std::to_array<const char *>(
			{"--gpu-architecture=compute_75", // Adjust for your GPU
			 "--use_fast_math",
			 "--std=c++17"}
		);
		e = nvrtcCompileProgram(prog, options.size(), options.data());
		if (e != NVRTC_SUCCESS) [[unlikely]] {
			log.write("x17ai_cuda_compile_kernel(): nvrtcCompileProgram() failed: ", e, "\n");

			// Get compilation log
			usize log_size = 0;
			e = nvrtcGetProgramLogSize(prog, &log_size);
			if (e != NVRTC_SUCCESS) [[unlikely]] {
				log.write(
					"x17ai_cuda_compile_kernel(): nvrtcGetProgramLogSize() failed: ", e, "\n"
				);
			} else {
				std::span log_span = log.extend(log_size);
				if (log_span.size() == log_size) [[likely]] {
					e = nvrtcGetProgramLog(prog, log_span.data());
					if (e != NVRTC_SUCCESS) [[unlikely]] {
						log.write(
							"x17ai_cuda_compile_kernel(): nvrtcGetProgramLog() failed: ",
							e, "\n"
						);
					}
				}
			}

			e = nvrtcDestroyProgram(&prog);
			if (e != NVRTC_SUCCESS) [[unlikely]] {
				log.write(
					"x17ai_cuda_compile_kernel(): nvrtcDestroyProgram() failed: ", e, "\n"
				);
			}

			return;
		}

		// Get PTX
		usize ptx_size = 0;
		e = nvrtcGetPTXSize(prog, &ptx_size);
		if (e != NVRTC_SUCCESS) [[unlikely]] {
			log.write("x17ai_cuda_compile_kernel(): nvrtcGetPTXSize() failed: ", e, "\n");
		} else if (ptx_size == 0) [[unlikely]] {
			log.write("x17ai_cuda_compile_kernel(): nvrtcGetPTXSize() returned size 0\n");
		} else {
			std::span ptx_span = ptx.extend(ptx_size + 1);
			if (ptx_span.size() != ptx_size + 1) [[unlikely]] {
				log.write("x17ai_cuda_compile_kernel(): failed to extend PTX buffer\n");
				ptx.clear();
			} else {
				e = nvrtcGetPTX(prog, ptx_span.data());
				ptx_span[ptx_size] = '\0';
				if (e != NVRTC_SUCCESS) [[unlikely]] {
					log.write("x17ai_cuda_compile_kernel(): nvrtcGetPTX() failed: ", e, "\n");
					ptx.clear();
				}
			}
		}

		e = nvrtcDestroyProgram(&prog);
		if (e != NVRTC_SUCCESS) [[unlikely]] {
			log.write("x17ai_cuda_compile_kernel(): nvrtcDestroyProgram() failed: ", e, "\n");
		}
	}

	CudaKernelHandle *x17ai_cuda_new_kernel(
		CudaContextHandle *context,
		char const *ptx,
		FfiBuffer err
	) noexcept {
		// TODO - need to also return CUfunction
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(context != nullptr);
			assert(ptx != nullptr);
			CUcontext ctx = reinterpret_cast<CUcontext>(context);
			CUresult e;

			e = cuCtxPushCurrent(ctx);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_new_kernel(): cuCtxPushCurrent() failed: ", e);
				return nullptr;
			}

			CUmodule module;
			e = cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);

			[[maybe_unused]] auto _e = cuCtxPopCurrent(&ctx);

			if (e != CUDA_SUCCESS) {
				err.write("x17ai_cuda_new_kernel(): cuModuleLoadDataEx() failed: ", e);
				return nullptr;
			}

			return reinterpret_cast<CudaKernelHandle *>(module);
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_new_kernel(): exception thrown: ", e.what());
			return nullptr;
		} catch (...) {
			err.write("x17ai_cuda_new_kernel(): unknown exception thrown");
			return nullptr;
		}
	}

	int x17ai_cuda_del_kernel(CudaKernelHandle *kernel, FfiBuffer err) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(kernel != nullptr);
			CUmodule module = reinterpret_cast<CUmodule>(kernel);
			CUresult e;

			e = cuModuleUnload(module);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_del_kernel(): cuModuleUnload() failed: ", e);
				return -1;
			}
			return 0;
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_del_kernel(): exception thrown: ", e.what());
			return -1;
		} catch (...) {
			err.write("x17ai_cuda_del_kernel(): unknown exception thrown");
			return -1;
		}
	}

	int x17ai_cuda_run_kernel(CudaKernelHandle *kernel,
		KernelOutput const *o,
		KernelElemArg const *elem_args,
		KernelReduceArg const *reduce_args,
		f64 const *const_args,
		FfiBuffer err
	) noexcept {
		// TODO
		return -1;
	}
}

/*
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
*/
