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
	void (*set_len)(void *self, usize new_len) noexcept;
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

	void set_len(usize new_len) noexcept {
		vmt->set_len(instance, new_len);
	}
};

// #[repr(C)]
// pub struct CudaCapability {
// 	pub major: usize,
// 	pub minor: usize,
// }

struct CudaCapability {
	usize major;
	usize minor;
};

struct CudaCube {
	usize x;
	usize y;
	usize z;
};

struct CudaLaunchConfig {
	CudaCube grid_dim;
	CudaCube block_dim;
	usize shared_mem_bytes;
};

struct CudaContextHandle {
	usize refcnt_munus_one;
	usize device_id;
	CudaCapability capability;
	usize warp_size;
	CUdevice device;
	CUcontext context;
};

struct CudaStreamHandle;
struct CudaModuleHandle;
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
	CudaContextHandle *x17ai_cuda_open_context(usize device_id, FfiBuffer err) noexcept {
		try {
			auto result = std::make_unique<CudaContextHandle>();
			result->refcnt_munus_one = 0;
			result->device_id = device_id;

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

			if (!std::in_range<int>(device_id)) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): CUDA device ID out of range");
				return nullptr;
			}

			e = cuDeviceGet(&result->device, int(unsigned(device_id)));
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): cuDeviceGet() failed: ", e);
				return nullptr;
			}

			int major = 0;
			e = cuDeviceGetAttribute(
				&major,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				result->device
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): cuDeviceGetAttribute(MAJOR) failed: ", e);
				return nullptr;
			}
			if (!std::in_range<usize>(major)) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): compute capability major version out of range");
				return nullptr;
			}
			result->capability.major = usize(major);

			int minor = 0;
			e = cuDeviceGetAttribute(
				&minor,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
				result->device
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): cuDeviceGetAttribute(MINOR) failed: ", e);
				return nullptr;
			}
			if (!std::in_range<usize>(minor)) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): compute capability minor version out of range");
				return nullptr;
			}
			result->capability.minor = usize(minor);

			int warp_size = 0;
			e = cuDeviceGetAttribute(
				&warp_size,
				CU_DEVICE_ATTRIBUTE_WARP_SIZE,
				result->device
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): cuDeviceGetAttribute(WARP_SIZE) failed: ", e);
				return nullptr;
			}
			if (!std::in_range<usize>(warp_size)) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): warp size out of range");
				return nullptr;
			}
			result->warp_size = usize(warp_size);
			if (!std::has_single_bit(result->warp_size)) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): warp size is not a power of two");
				return nullptr;
			}

			e = cuDevicePrimaryCtxRetain(&result->context, result->device);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() failed: ", e);
				return nullptr;
			}
			if (result->context == nullptr) [[unlikely]] {
				err.write("x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() returned nullptr");
				return nullptr;
			}

			return result.release();
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_open_context(): exception thrown: ", e.what());
			return nullptr;
		} catch (...) {
			err.write("x17ai_cuda_open_context(): unknown exception thrown");
			return nullptr;
		}
	}

	int x17ai_cuda_close_context(CudaContextHandle *context, FfiBuffer err) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(context != nullptr);
			if (context->refcnt_munus_one > 0) {
				--context->refcnt_munus_one;
				return 0;
			}
			auto ctx = std::unique_ptr<CudaContextHandle>(context);
			CUresult e;

			e = cuDevicePrimaryCtxRelease(ctx->device);
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

	CudaStreamHandle *x17ai_cuda_open_stream(CudaContextHandle *ctx, FfiBuffer err) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(ctx != nullptr);
			CUresult e;

			e = cuCtxPushCurrent(ctx->context);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_open_stream(): cuCtxPushCurrent() failed: ", e);
				return nullptr;
			}

			CUstream cu_stream;
			e = cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING);

			CUcontext popped_ctx;
			[[maybe_unused]] auto _e = cuCtxPopCurrent(&popped_ctx);

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

	void x17ai_cuda_compile_module(
		CudaCapability device_capability,
		char const *source,
		FfiBuffer ptx,
		FfiBuffer log
	) noexcept {
		try {
			nvrtcResult e;

			std::array<char, 40> arch_opt{};
			auto t = fmt::format_to_n(
				arch_opt.data(),
				arch_opt.size(),
				"--gpu-architecture=compute_{}{}",
				device_capability.major,
				device_capability.minor
			);
			if (t.size >= arch_opt.size()) [[unlikely]] {
				log.write("x17ai_cuda_compile_module(): architecture string too long\n");
				return;
			}
			arch_opt[t.size] = '\0';

			auto options = std::to_array<const char *>({
				arch_opt.data(),
				"--std=c++17",
				"--use_fast_math",
			});

			// Create NVRTC program
			nvrtcProgram prog;
			e = nvrtcCreateProgram(&prog, source, "module.cu", 0, nullptr, nullptr);
			if (e != NVRTC_SUCCESS) [[unlikely]] {
				log.write("x17ai_cuda_compile_module(): nvrtcCreateProgram() failed: ", e);
				return;
			}

			// Compile the program
			e = nvrtcCompileProgram(prog, options.size(), options.data());
			if (e != NVRTC_SUCCESS) [[unlikely]] {
				log.write("x17ai_cuda_compile_module(): nvrtcCompileProgram() failed: ", e, "\n");

				// Get compilation log
				usize log_size = 0;
				e = nvrtcGetProgramLogSize(prog, &log_size);
				if (e != NVRTC_SUCCESS) [[unlikely]] {
					log.write(
						"x17ai_cuda_compile_module(): nvrtcGetProgramLogSize() failed: ", e, "\n"
					);
				} else {
					std::span log_span = log.extend(log_size);
					if (log_span.size() == log_size) [[likely]] {
						e = nvrtcGetProgramLog(prog, log_span.data());
						if (e != NVRTC_SUCCESS) [[unlikely]] {
							log.write(
								"x17ai_cuda_compile_module(): nvrtcGetProgramLog() failed: ",
								e, "\n"
							);
						}
					}
				}

				e = nvrtcDestroyProgram(&prog);
				if (e != NVRTC_SUCCESS) [[unlikely]] {
					log.write(
						"x17ai_cuda_compile_module(): nvrtcDestroyProgram() failed: ", e, "\n"
					);
				}

				return;
			}

			// Get PTX
			usize ptx_size = 0;
			e = nvrtcGetPTXSize(prog, &ptx_size);
			if (e != NVRTC_SUCCESS) [[unlikely]] {
				log.write("x17ai_cuda_compile_module(): nvrtcGetPTXSize() failed: ", e, "\n");
			} else if (ptx_size == 0) [[unlikely]] {
				log.write("x17ai_cuda_compile_module(): nvrtcGetPTXSize() returned size 0\n");
			} else {
				std::span ptx_span = ptx.extend(ptx_size + 1);
				if (ptx_span.size() != ptx_size + 1) [[unlikely]] {
					log.write("x17ai_cuda_compile_module(): failed to extend PTX buffer\n");
					ptx.clear();
				} else {
					e = nvrtcGetPTX(prog, ptx_span.data());
					ptx_span[ptx_size] = '\0';
					ptx.set_len(ptx_size);
					if (e != NVRTC_SUCCESS) [[unlikely]] {
						log.write("x17ai_cuda_compile_module(): nvrtcGetPTX() failed: ", e, "\n");
						ptx.clear();
					}
				}
			}

			e = nvrtcDestroyProgram(&prog);
			if (e != NVRTC_SUCCESS) [[unlikely]] {
				log.write("x17ai_cuda_compile_module(): nvrtcDestroyProgram() failed: ", e, "\n");
			}
		} catch (std::exception const &e) {
			log.write("x17ai_cuda_compile_module(): exception thrown: ", e.what());
			ptx.clear();
		} catch (...) {
			log.write("x17ai_cuda_compile_module(): unknown exception thrown");
			ptx.clear();
		}
	}

	CudaModuleHandle *x17ai_cuda_load_module(
		CudaContextHandle *ctx,
		char const *ptx,
		FfiBuffer err
	) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(ctx != nullptr);
			assert(ptx != nullptr);
			CUresult e;

			e = cuCtxPushCurrent(ctx->context);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_load_module(): cuCtxPushCurrent() failed: ", e);
				return nullptr;
			}

			CUmodule module;
			e = cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);

			CUcontext popped_ctx;
			[[maybe_unused]] auto _e = cuCtxPopCurrent(&popped_ctx);

			if (e != CUDA_SUCCESS) {
				err.write("x17ai_cuda_load_module(): cuModuleLoadDataEx() failed: ", e);
				return nullptr;
			}

			return reinterpret_cast<CudaModuleHandle *>(module);
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_load_module(): exception thrown: ", e.what());
			return nullptr;
		} catch (...) {
			err.write("x17ai_cuda_load_module(): unknown exception thrown");
			return nullptr;
		}
	}

	int x17ai_cuda_del_module(CudaModuleHandle *mod, FfiBuffer err) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(mod != nullptr);
			CUmodule module = reinterpret_cast<CUmodule>(mod);
			CUresult e;

			e = cuModuleUnload(module);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_del_module(): cuModuleUnload() failed: ", e);
				return -1;
			}
			return 0;
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_del_module(): exception thrown: ", e.what());
			return -1;
		} catch (...) {
			err.write("x17ai_cuda_del_module(): unknown exception thrown");
			return -1;
		}
	}

	CudaKernelHandle *x17ai_cuda_get_kernel(
		CudaModuleHandle *mod,
		char const *name,
		FfiBuffer err
	) {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(mod != nullptr);
			assert(name != nullptr);
			CUmodule module = reinterpret_cast<CUmodule>(mod);
			CUresult e;

			CUfunction kernel;
			e = cuModuleGetFunction(&kernel, module, name);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_get_kernel(): cuModuleGetFunction() failed: ", e);
				return nullptr;
			}
			return reinterpret_cast<CudaKernelHandle *>(kernel);
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_get_kernel(): exception thrown: ", e.what());
			return nullptr;
		} catch (...) {
			err.write("x17ai_cuda_get_kernel(): unknown exception thrown");
			return nullptr;
		}
	}

	int x17ai_cuda_run_kernel(
		CudaStreamHandle *stream,
		CudaKernelHandle *kernel,
		CudaLaunchConfig const *config,
		void const* const*args,
		FfiBuffer err
	) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			assert(kernel != nullptr);
			assert(config != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUfunction function = reinterpret_cast<CUfunction>(kernel);
			CUresult result;

			size_t t =
				config->block_dim.x
				| config->block_dim.y
				| config->block_dim.z
				| config->grid_dim.x
				| config->grid_dim.y
				| config->grid_dim.z
				| config->shared_mem_bytes;
			if (!std::in_range<unsigned int>(t)) [[unlikely]] {
				err.write("x17ai_cuda_run_kernel(): launch configuration values out of range");
				return -1;
			}

			result = cuLaunchKernel(
				function,
				static_cast<unsigned int>(config->grid_dim.x),
				static_cast<unsigned int>(config->grid_dim.y),
				static_cast<unsigned int>(config->grid_dim.z),
				static_cast<unsigned int>(config->block_dim.x),
				static_cast<unsigned int>(config->block_dim.y),
				static_cast<unsigned int>(config->block_dim.z),
				static_cast<unsigned int>(config->shared_mem_bytes),
				cu_stream,
				const_cast<void **>(args),
				nullptr
			);
			if (result != CUDA_SUCCESS) [[unlikely]] {
				err.write("x17ai_cuda_run_kernel(): cuLaunchKernel() failed: ", result);
				return -1;
			}
			return 0;
		} catch (std::exception const &e) {
			err.write("x17ai_cuda_run_kernel(): exception thrown: ", e.what());
			return -1;
		} catch (...) {
			err.write("x17ai_cuda_run_kernel(): unknown exception thrown");
			return -1;
		}
	}
}
