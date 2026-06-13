#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <atomic>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <string_view>
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
	size_t len;
};

struct FfiBufferVMT {
	FfiSpan (*span)(void *self) noexcept;
	FfiSpan (*buf_span)(void *self) noexcept;
	FfiSpan (*extend)(void *self, size_t additional) noexcept;
	void (*clear)(void *self) noexcept;
	void (*set_len)(void *self, size_t new_len) noexcept;
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

	inline std::span<char> extend(size_t additional) noexcept {
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

	void set_len(size_t new_len) noexcept {
		vmt->set_len(instance, new_len);
	}
};

struct DiagnosticBuffer {
	std::string message;

	DiagnosticBuffer():
		message()
	{}

	bool write() noexcept {
		return true;
	}

	bool write(std::string_view str) noexcept {
		try {
			message.append(str);
			return true;
		} catch (...) {
			return false;
		}
	}

	bool write(int value) noexcept {
		try {
			return write(fmt::to_string(value));
		} catch (...) {
			return false;
		}
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
};

template<typename... TS>
std::unique_ptr<DiagnosticBuffer> new_diag(TS const &... ts) noexcept {
	std::unique_ptr<DiagnosticBuffer> result;
	try {
		result = std::make_unique<DiagnosticBuffer>();
		result->write(ts...);
	} catch (...) {}
	return result;
}

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

inline void assert_no_context() {
	#ifndef NDEBUG
	CUcontext pctx;
	cuCtxGetCurrent(&pctx);
	if (pctx != nullptr) {
		throw "we have context!";
	}
	#endif
}

template<typename T>
struct PtrResult {
	T *value;
	DiagnosticBuffer *diagnostic;

	PtrResult(T *val):
		value(val),
		diagnostic(nullptr)
	{}

	PtrResult(std::unique_ptr<DiagnosticBuffer> diag):
		value(nullptr),
		diagnostic(diag.release())
	{}
};

struct UsizeResult {
	size_t value;
	DiagnosticBuffer *diagnostic;

	UsizeResult():
		value(0),
		diagnostic(nullptr)
	{}

	UsizeResult(size_t val, std::unique_ptr<DiagnosticBuffer> diag):
		value(val == 0 ? size_t(-1) : val),
		diagnostic(diag.release())
	{}

	UsizeResult(std::unique_ptr<DiagnosticBuffer> diag):
		value(-1),
		diagnostic(diag.release())
	{}
};

extern "C" {
	void x17ai_copy_diagnostic(DiagnosticBuffer *diag, FfiBuffer err) noexcept {
		if (diag == nullptr) {
			return;
		}

		try {
			auto diag_owner = std::unique_ptr<DiagnosticBuffer>(diag);
			auto const &message = diag_owner->message;
			err.write(std::string_view(message.data(), message.size()));
		} catch (...) {}
	}

	PtrResult<CudaContextHandle> x17ai_cuda_open_context(usize device_id) noexcept {
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
						return new_diag("x17ai_cuda_open_context(): cuInit() failed: ", e);
					}
					cuda_initialized.store(true, std::memory_order_release);
				}
			}

			if (!std::in_range<int>(device_id)) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): CUDA device ID out of range");
			}

			e = cuDeviceGet(&result->device, int(unsigned(device_id)));
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): cuDeviceGet() failed: ", e);
			}

			int major = 0;
			e = cuDeviceGetAttribute(
				&major,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				result->device
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): cuDeviceGetAttribute(MAJOR) failed: ", e);
			}
			if (!std::in_range<usize>(major)) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): compute capability major version out of range");
			}
			result->capability.major = usize(major);

			int minor = 0;
			e = cuDeviceGetAttribute(
				&minor,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
				result->device
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): cuDeviceGetAttribute(MINOR) failed: ", e);
			}
			if (!std::in_range<usize>(minor)) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): compute capability minor version out of range");
			}
			result->capability.minor = usize(minor);

			int warp_size = 0;
			e = cuDeviceGetAttribute(
				&warp_size,
				CU_DEVICE_ATTRIBUTE_WARP_SIZE,
				result->device
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): cuDeviceGetAttribute(WARP_SIZE) failed: ", e);
			}
			if (!std::in_range<usize>(warp_size)) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): warp size out of range");
			}
			result->warp_size = usize(warp_size);
			if (!std::has_single_bit(result->warp_size)) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): warp size is not a power of two");
			}

			e = cuDevicePrimaryCtxRetain(&result->context, result->device);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() failed: ", e);
			}
			if (result->context == nullptr) [[unlikely]] {
				return new_diag("x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() returned nullptr");
			}

			return result.release();
		} catch (std::exception const &e) {
			return new_diag("x17ai_cuda_open_context(): exception thrown: ", e.what());
		} catch (...) {
			return new_diag("x17ai_cuda_open_context(): unknown exception thrown");
		}
	}

	UsizeResult x17ai_cuda_close_context(CudaContextHandle *context) noexcept {
		try {
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(context != nullptr);
			if (context->refcnt_munus_one > 0) {
				--context->refcnt_munus_one;
				return UsizeResult();
			}
			auto ctx = std::unique_ptr<CudaContextHandle>(context);
			CUresult e;

			e = cuDevicePrimaryCtxRelease(ctx->device);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return UsizeResult(e,
					new_diag("x17ai_cuda_close_context(): cuDevicePrimaryCtxRelease() failed: ", e)
				);
			}
			return UsizeResult();
		} catch (std::exception const &e) {
			return new_diag("x17ai_cuda_close_context(): exception thrown: ", e.what());
		} catch (...) {
			return new_diag("x17ai_cuda_close_context(): unknown exception thrown");
		}
	}

	void *x17ai_cuda_context_ptr(CudaContextHandle *context) noexcept {
		assert_no_context();
		assert(cuda_initialized.load(std::memory_order_acquire));
		assert(context != nullptr);
		return static_cast<void *>(context->context);
	}

	PtrResult<CudaStreamHandle> x17ai_cuda_open_stream(CudaContextHandle *ctx) noexcept {
		try {
			assert_no_context();
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(ctx != nullptr);
			CUresult e;

			e = cuCtxPushCurrent(ctx->context);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return new_diag("x17ai_cuda_open_stream(): cuCtxPushCurrent() failed: ", e);
			}

			CUstream cu_stream;
			e = cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING);

			CUcontext popped_ctx;
			[[maybe_unused]] auto _e = cuCtxPopCurrent(&popped_ctx);

			if (e != CUDA_SUCCESS) [[unlikely]] {
				return new_diag("x17ai_cuda_open_stream(): cuStreamCreate() failed: ", e);
			}
			if (cu_stream == nullptr) [[unlikely]] {
				return new_diag("x17ai_cuda_open_stream(): cuStreamCreate() returned nullptr");
			}

			assert_no_context();
			return reinterpret_cast<CudaStreamHandle *>(cu_stream);
		} catch (std::exception const &e) {
			return new_diag("x17ai_cuda_open_stream(): exception thrown: ", e.what());
		} catch (...) {
			return new_diag("x17ai_cuda_open_stream(): unknown exception thrown");
		}
	}

	UsizeResult x17ai_cuda_close_stream(CudaStreamHandle *stream) noexcept {
		try {
			assert_no_context();
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuStreamDestroy(cu_stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return UsizeResult(e,
					new_diag("x17ai_cuda_close_stream(): cuStreamDestroy() failed: ", e)
				);
			}

			assert_no_context();
			return UsizeResult();
		} catch (std::exception const &e) {
			return new_diag("x17ai_cuda_close_stream(): exception thrown: ", e.what());
		} catch (...) {
			return new_diag("x17ai_cuda_close_stream(): unknown exception thrown");
		}
	}

	UsizeResult x17ai_cuda_synchronize(CudaStreamHandle *stream) noexcept {
		try {
			assert_no_context();
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuStreamSynchronize(cu_stream);

			if (e != CUDA_SUCCESS) [[unlikely]] {
				return UsizeResult(e,
					new_diag("x17ai_cuda_synchronize(): cuStreamSynchronize() failed: ", e)
				);
			}

			assert_no_context();
			return UsizeResult();
		} catch (std::exception const &e) {
			return new_diag("x17ai_cuda_synchronize(): exception thrown: ", e.what());
		} catch (...) {
			return new_diag("x17ai_cuda_synchronize(): unknown exception thrown");
		}
	}

	PtrResult<CudaDeviceData> x17ai_cuda_alloc(CudaStreamHandle *stream, usize bytes) noexcept {
		try {
			assert_no_context();
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			CUdeviceptr memory = 0;
			e = cuMemAllocAsync(&memory, bytes, cu_stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return new_diag("x17ai_cuda_alloc(): cuMemAllocAsync() failed: ", e);
			}
			if (memory == 0) [[unlikely]] {
				return new_diag("x17ai_cuda_alloc(): cuMemAllocAsync() returned null pointer");
			}

			assert_no_context();
			return from_dev_ptr(memory);
		} catch (std::exception const &e) {
			return new_diag("x17ai_cuda_alloc(): exception thrown: ", e.what());
		} catch (...) {
			return new_diag("x17ai_cuda_alloc(): unknown exception thrown");
		}
	}

	UsizeResult x17ai_cuda_free(CudaStreamHandle *stream, CudaDeviceData *ptr) noexcept {
		try {
			assert_no_context();
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuMemFreeAsync(to_dev_ptr(ptr), cu_stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return UsizeResult(e,
					new_diag("x17ai_cuda_free(): cuMemFreeAsync() failed: ", e)
				);
			}

			assert_no_context();
			return UsizeResult();
		} catch (std::exception const &e) {
			return new_diag("x17ai_cuda_free(): exception thrown: ", e.what());
		} catch (...) {
			return new_diag("x17ai_cuda_free(): unknown exception thrown");
		}
	}

	UsizeResult x17ai_cuda_upload_data(
		CudaStreamHandle *stream,
		u8 const *src,
		CudaDeviceData *dst,
		usize offset_bytes,
		usize size_bytes
	) noexcept {
		try {
			assert_no_context();
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
				return UsizeResult(e,
					new_diag("x17ai_cuda_upload_data(): cuMemcpyHtoDAsync() failed: ", e)
				);
			}

			assert_no_context();
			return UsizeResult();
		} catch (std::exception const &e) {
			return new_diag("x17ai_cuda_upload_data(): exception thrown: ", e.what());
		} catch (...) {
			return new_diag("x17ai_cuda_upload_data(): unknown exception thrown");
		}
	}

	UsizeResult x17ai_cuda_download_data(
		CudaStreamHandle *stream,
		CudaDeviceData *src,
		u8 *dst,
		usize offset_bytes,
		usize size_bytes
	) noexcept {
		try {
			assert_no_context();
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
				return UsizeResult(e,
					new_diag("x17ai_cuda_download_data(): cuMemcpyDtoHAsync() failed: ", e)
				);
			}

			assert_no_context();
			return UsizeResult();
		} catch (std::exception const &e) {
			return new_diag("x17ai_cuda_download_data(): exception thrown: ", e.what());
		} catch (...) {
			return new_diag("x17ai_cuda_download_data(): unknown exception thrown");
		}
	}
}
