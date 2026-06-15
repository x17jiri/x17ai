#include <cuda.h>
#include <atomic>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <unordered_map>
#include <cassert>
#include <span>
#include <vector>
#include <dlfcn.h>
#include <utility>

#include "ffi.hpp"

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

struct CudaTimerHandle {
	CUcontext context;
	CUstream stream;
	CUevent start;
	CUevent end;
};

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

inline void destroy_timer_events(CudaTimerHandle *timer) noexcept {
	if (timer->end != nullptr) {
		[[maybe_unused]] CUresult ignored = cuEventDestroy(timer->end);
		timer->end = nullptr;
	}
	if (timer->start != nullptr) {
		[[maybe_unused]] CUresult ignored = cuEventDestroy(timer->start);
		timer->start = nullptr;
	}
}

extern "C" {
	void x17ai_move_diagnostic(DiagnosticBuffer *diag, FfiBuffer err) noexcept {
		if (diag == nullptr) {
			return;
		}

		try {
			std::unique_ptr<AllocatedDiagnosticBuffer> owner;
			if (diag->is_allocated) {
				owner = std::unique_ptr<AllocatedDiagnosticBuffer>(
					static_cast<AllocatedDiagnosticBuffer *>(diag)
				);
			}
			auto const &message = diag->message;
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
						return X17AI_DIAG(
							"x17ai_cuda_open_context(): cuInit() failed.",
							"x17ai_cuda_open_context(): cuInit() failed with cuda error: ", e
						);
					}
					cuda_initialized.store(true, std::memory_order_release);
				}
			}

			if (!std::in_range<int>(device_id)) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_open_context(): CUDA device ID out of range"
				);
			}

			e = cuDeviceGet(&result->device, int(unsigned(device_id)));
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_open_context(): cuDeviceGet() failed.",
					"x17ai_cuda_open_context(): cuDeviceGet() failed with cuda error: ", e
				);
			}

			int major = 0;
			e = cuDeviceGetAttribute(
				&major,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				result->device
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_open_context(): cuDeviceGetAttribute(MAJOR) failed.",
					"x17ai_cuda_open_context(): cuDeviceGetAttribute(MAJOR) failed with cuda error: ", e
				);
			}
			if (!std::in_range<usize>(major)) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_open_context(): compute capability major version out of range"
				);
			}
			result->capability.major = usize(major);

			int minor = 0;
			e = cuDeviceGetAttribute(
				&minor,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
				result->device
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_open_context(): cuDeviceGetAttribute(MINOR) failed.",
					"x17ai_cuda_open_context(): cuDeviceGetAttribute(MINOR) failed with cuda error: ", e
				);
			}
			if (!std::in_range<usize>(minor)) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_open_context(): compute capability minor version out of range"
				);
			}
			result->capability.minor = usize(minor);

			int warp_size = 0;
			e = cuDeviceGetAttribute(
				&warp_size,
				CU_DEVICE_ATTRIBUTE_WARP_SIZE,
				result->device
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_open_context(): cuDeviceGetAttribute(WARP_SIZE) failed.",
					"x17ai_cuda_open_context(): cuDeviceGetAttribute(WARP_SIZE) failed with cuda error: ", e
				);
			}
			if (!std::in_range<usize>(warp_size)) [[unlikely]] {
				return X17AI_STATIC_DIAG("x17ai_cuda_open_context(): warp size out of range");
			}
			result->warp_size = usize(warp_size);
			if (!std::has_single_bit(result->warp_size)) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_open_context(): warp size is not a power of two"
				);
			}

			e = cuDevicePrimaryCtxRetain(&result->context, result->device);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() failed.",
					"x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() failed with cuda error: ", e
				);
			}
			if (result->context == nullptr) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_open_context(): cuDevicePrimaryCtxRetain() returned nullptr"
				);
			}

			return result.release();
		} X17AI_CATCH_ERRORS("x17ai_cuda_open_context()")
	}

	DiagnosticBuffer *x17ai_cuda_close_context(CudaContextHandle *context) noexcept {
		try {
			assert_no_context("x17ai_cuda_close_context(): before cuDevicePrimaryCtxRelease()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(context != nullptr);
			if (context->refcnt_munus_one > 0) {
				--context->refcnt_munus_one;
				return nullptr;
			}
			auto ctx = std::unique_ptr<CudaContextHandle>(context);
			CUresult e;

			e = cuDevicePrimaryCtxRelease(ctx->device);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_close_context(): cuDevicePrimaryCtxRelease() failed.",
					"x17ai_cuda_close_context(): cuDevicePrimaryCtxRelease() failed with cuda error: ", e
				);
			}
			assert_no_context("x17ai_cuda_close_context(): after cuDevicePrimaryCtxRelease()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_close_context()")
	}

	void *x17ai_cuda_context_ptr(CudaContextHandle *context) noexcept {
		assert(cuda_initialized.load(std::memory_order_acquire));
		assert(context != nullptr);
		return static_cast<void *>(context->context);
	}

	PtrResult<CudaStreamHandle> x17ai_cuda_open_stream(CudaContextHandle *ctx) noexcept {
		try {
			assert_no_context("x17ai_cuda_open_stream(): before cuCtxPushCurrent()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(ctx != nullptr);
			CUresult e;

			e = cuCtxPushCurrent(ctx->context);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_open_stream(): cuCtxPushCurrent() failed.",
					"x17ai_cuda_open_stream(): cuCtxPushCurrent() failed with cuda error: ", e
				);
			}

			CUstream cu_stream;
			e = cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING);

			CUcontext popped_ctx;
			// Ignore `_e`. I think `pop` at this point shouldn't fail.
			// If we ever decide to handle this error, we need to free the stream we just created.
			[[maybe_unused]] auto _e = cuCtxPopCurrent(&popped_ctx);

			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_open_stream(): cuStreamCreate() failed.",
					"x17ai_cuda_open_stream(): cuStreamCreate() failed with cuda error: ", e
				);
			}
			if (cu_stream == nullptr) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_open_stream(): cuStreamCreate() returned nullptr"
				);
			}

			assert_no_context("x17ai_cuda_open_stream(): after cuCtxPopCurrent()");
			return reinterpret_cast<CudaStreamHandle *>(cu_stream);
		} X17AI_CATCH_ERRORS("x17ai_cuda_open_stream()")
	}

	DiagnosticBuffer *x17ai_cuda_close_stream(CudaStreamHandle *stream) noexcept {
		try {
			assert_no_context("x17ai_cuda_close_stream(): before cuStreamDestroy()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuStreamDestroy(cu_stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_close_stream(): cuStreamDestroy() failed.",
					"x17ai_cuda_close_stream(): cuStreamDestroy() failed with cuda error: ", e
				);
			}

			assert_no_context("x17ai_cuda_close_stream(): after cuStreamDestroy()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_close_stream()")
	}

	DiagnosticBuffer *x17ai_cuda_synchronize(CudaStreamHandle *stream) noexcept {
		try {
			assert_no_context("x17ai_cuda_synchronize(): before cuStreamSynchronize()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuStreamSynchronize(cu_stream);

			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_synchronize(): cuStreamSynchronize() failed.",
					"x17ai_cuda_synchronize(): cuStreamSynchronize() failed with cuda error: ", e
				);
			}

			assert_no_context("x17ai_cuda_synchronize(): after cuStreamSynchronize()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_synchronize()")
	}

	PtrResult<CudaTimerHandle> x17ai_cuda_create_timer(CudaContextHandle *context, CudaStreamHandle *stream) noexcept {
		try {
			assert_no_context("x17ai_cuda_create_timer(): before cuCtxPushCurrent()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(context != nullptr);
			assert(stream != nullptr);
			CUcontext cu_context = context->context;
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);

			auto result = std::make_unique<CudaTimerHandle>();
			result->context =cu_context;
			result->stream = cu_stream;
			result->start = nullptr;
			result->end = nullptr;

			CUresult e = cuCtxPushCurrent(cu_context);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_create_timer(): cuCtxPushCurrent() failed.",
					"x17ai_cuda_create_timer(): cuCtxPushCurrent() failed with cuda error: ", e
				);
			}

			DiagnosticBuffer *diagnostic = nullptr;
			e = cuEventCreate(&result->start, CU_EVENT_DEFAULT);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				diagnostic = X17AI_DIAG(
					"x17ai_cuda_create_timer(): cuEventCreate(start) failed.",
					"x17ai_cuda_create_timer(): cuEventCreate(start) failed with cuda error: ", e
				);
			}

			if (diagnostic == nullptr) {
				e = cuEventCreate(&result->end, CU_EVENT_DEFAULT);
				if (e != CUDA_SUCCESS) [[unlikely]] {
					diagnostic = X17AI_DIAG(
						"x17ai_cuda_create_timer(): cuEventCreate(end) failed.",
						"x17ai_cuda_create_timer(): cuEventCreate(end) failed with cuda error: ", e
					);
				}
			}

			if (diagnostic != nullptr) {
				destroy_timer_events(result.get());
			}

			CUcontext popped_ctx = nullptr;
			CUresult pop_err = cuCtxPopCurrent(&popped_ctx);
			if (pop_err != CUDA_SUCCESS) [[unlikely]] {
				if (diagnostic == nullptr) {
					destroy_timer_events(result.get());
					diagnostic = X17AI_DIAG(
						"x17ai_cuda_create_timer(): cuCtxPopCurrent() failed.",
						"x17ai_cuda_create_timer(): cuCtxPopCurrent() failed with cuda error: ", pop_err
					);
				}
				return diagnostic;
			}
			if (diagnostic != nullptr) {
				return diagnostic;
			}
			if (popped_ctx != cu_context) [[unlikely]] {
				destroy_timer_events(result.get());
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_create_timer(): popped unexpected CUDA context"
				);
			}

			assert_no_context("x17ai_cuda_create_timer(): after cuCtxPopCurrent()");
			return result.release();
		} X17AI_CATCH_ERRORS("x17ai_cuda_create_timer()")
	}

	DiagnosticBuffer *x17ai_cuda_destroy_timer(CudaTimerHandle *timer) noexcept {
		try {
			assert_no_context("x17ai_cuda_destroy_timer(): before cuCtxPushCurrent()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(timer != nullptr);
			auto timer_owner = std::unique_ptr<CudaTimerHandle>(timer);

			CUresult e = cuCtxPushCurrent(timer->context);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_destroy_timer(): cuCtxPushCurrent() failed.",
					"x17ai_cuda_destroy_timer(): cuCtxPushCurrent() failed with cuda error: ", e
				);
			}

			DiagnosticBuffer *diagnostic = nullptr;
			if (timer->end != nullptr) {
				e = cuEventDestroy(timer->end);
				timer->end = nullptr;
				if (e != CUDA_SUCCESS) [[unlikely]] {
					diagnostic = X17AI_DIAG(
						"x17ai_cuda_destroy_timer(): cuEventDestroy(end) failed.",
						"x17ai_cuda_destroy_timer(): cuEventDestroy(end) failed with cuda error: ", e
					);
				}
			}
			if (timer->start != nullptr) {
				e = cuEventDestroy(timer->start);
				timer->start = nullptr;
				if (e != CUDA_SUCCESS && diagnostic == nullptr) [[unlikely]] {
					diagnostic = X17AI_DIAG(
						"x17ai_cuda_destroy_timer(): cuEventDestroy(start) failed.",
						"x17ai_cuda_destroy_timer(): cuEventDestroy(start) failed with cuda error: ", e
					);
				}
			}

			CUcontext popped_ctx = nullptr;
			CUresult pop_err = cuCtxPopCurrent(&popped_ctx);
			if (diagnostic != nullptr) {
				return diagnostic;
			}
			if (pop_err != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_destroy_timer(): cuCtxPopCurrent() failed.",
					"x17ai_cuda_destroy_timer(): cuCtxPopCurrent() failed with cuda error: ", pop_err
				);
			}
			if (popped_ctx != timer->context) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_destroy_timer(): popped unexpected CUDA context"
				);
			}

			assert_no_context("x17ai_cuda_destroy_timer(): after cuCtxPopCurrent()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_destroy_timer()")
	}

	DiagnosticBuffer *x17ai_cuda_timer_start(CudaTimerHandle *timer) noexcept {
		try {
			assert_no_context("x17ai_cuda_timer_start(): before cuEventRecord()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(timer != nullptr);

			CUresult e = cuEventRecord(timer->start, timer->stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_timer_start(): cuEventRecord() failed.",
					"x17ai_cuda_timer_start(): cuEventRecord() failed with cuda error: ", e
				);
			}

			assert_no_context("x17ai_cuda_timer_start(): after cuEventRecord()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_timer_start()")
	}

	DiagnosticBuffer *x17ai_cuda_timer_stop(CudaTimerHandle *timer) noexcept {
		try {
			assert_no_context("x17ai_cuda_timer_stop(): before cuEventRecord()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(timer != nullptr);

			CUresult e = cuEventRecord(timer->end, timer->stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_timer_stop(): cuEventRecord() failed.",
					"x17ai_cuda_timer_stop(): cuEventRecord() failed with cuda error: ", e
				);
			}

			assert_no_context("x17ai_cuda_timer_stop(): after cuEventRecord()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_timer_stop()")
	}

	DiagnosticBuffer *x17ai_cuda_timer_elapsed_seconds(
		CudaTimerHandle *timer,
		f64 *seconds
	) noexcept {
		try {
			assert_no_context("x17ai_cuda_timer_elapsed_seconds(): before cuCtxPushCurrent()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(timer != nullptr);
			assert(seconds != nullptr);

			CUresult e = cuCtxPushCurrent(timer->context);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_timer_elapsed_seconds(): cuCtxPushCurrent() failed.",
					"x17ai_cuda_timer_elapsed_seconds(): cuCtxPushCurrent() failed with cuda error: ", e
				);
			}

			DiagnosticBuffer *diagnostic = nullptr;
			f32 ms = 0.0;
			e = cuEventElapsedTime(&ms, timer->start, timer->end);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				diagnostic = X17AI_DIAG(
					"x17ai_cuda_timer_elapsed_seconds(): cuEventElapsedTime() failed.",
					"x17ai_cuda_timer_elapsed_seconds(): cuEventElapsedTime() failed with cuda error: ", e
				);
			}

			CUcontext popped_ctx = nullptr;
			CUresult pop_err = cuCtxPopCurrent(&popped_ctx);
			if (diagnostic != nullptr) {
				return diagnostic;
			}
			if (pop_err != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_timer_elapsed_seconds(): cuCtxPopCurrent() failed.",
					"x17ai_cuda_timer_elapsed_seconds(): cuCtxPopCurrent() failed with cuda error: ", pop_err
				);
			}
			if (popped_ctx != timer->context) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_timer_elapsed_seconds(): popped unexpected CUDA context"
				);
			}

			assert_no_context("x17ai_cuda_timer_elapsed_seconds(): after cuCtxPopCurrent()");

			*seconds = f64(ms) / 1000.0;
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_timer_elapsed_seconds()")
	}

	PtrResult<CudaDeviceData> x17ai_cuda_alloc(CudaStreamHandle *stream, usize bytes) noexcept {
		try {
			assert_no_context("x17ai_cuda_alloc(): before cuMemAllocAsync()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			CUdeviceptr memory = 0;
			e = cuMemAllocAsync(&memory, bytes, cu_stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_alloc(): cuMemAllocAsync() failed.",
					"x17ai_cuda_alloc(): cuMemAllocAsync() failed with cuda error: ", e
				);
			}
			if (memory == 0) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_alloc(): cuMemAllocAsync() returned null pointer"
				);
			}

			assert_no_context("x17ai_cuda_alloc(): after cuMemAllocAsync()");
			return from_dev_ptr(memory);
		} X17AI_CATCH_ERRORS("x17ai_cuda_alloc()")
	}

	DiagnosticBuffer *x17ai_cuda_free(CudaStreamHandle *stream, CudaDeviceData *ptr) noexcept {
		try {
			assert_no_context("x17ai_cuda_free(): before cuMemFreeAsync()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUresult e;

			e = cuMemFreeAsync(to_dev_ptr(ptr), cu_stream);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_free(): cuMemFreeAsync() failed.",
					"x17ai_cuda_free(): cuMemFreeAsync() failed with cuda error: ", e
				);
			}

			assert_no_context("x17ai_cuda_free(): after cuMemFreeAsync()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_free()")
	}

	DiagnosticBuffer *x17ai_cuda_upload_data(
		CudaStreamHandle *stream,
		u8 const *src,
		CudaDeviceData *dst,
		usize offset_bytes,
		usize size_bytes
	) noexcept {
		try {
			assert_no_context("x17ai_cuda_upload_data(): before cuMemcpyHtoDAsync()");
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
				return X17AI_DIAG(
					"x17ai_cuda_upload_data(): cuMemcpyHtoDAsync() failed.",
					"x17ai_cuda_upload_data(): cuMemcpyHtoDAsync() failed with cuda error: ", e
				);
			}

			assert_no_context("x17ai_cuda_upload_data(): after cuMemcpyHtoDAsync()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_upload_data()")
	}

	DiagnosticBuffer *x17ai_cuda_download_data(
		CudaStreamHandle *stream,
		CudaDeviceData *src,
		u8 *dst,
		usize offset_bytes,
		usize size_bytes
	) noexcept {
		try {
			assert_no_context("x17ai_cuda_download_data(): before cuMemcpyDtoHAsync()");
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
				return X17AI_DIAG(
					"x17ai_cuda_download_data(): cuMemcpyDtoHAsync() failed.",
					"x17ai_cuda_download_data(): cuMemcpyDtoHAsync() failed with cuda error: ", e
				);
			}

			assert_no_context("x17ai_cuda_download_data(): after cuMemcpyDtoHAsync()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_download_data()")
	}

	PtrResult<CudaModuleHandle> x17ai_cuda_load_module(
		CudaContextHandle *ctx,
		char const *cubin_path
	) noexcept {
		try {
			assert_no_context("x17ai_cuda_load_module(): before cuCtxPushCurrent()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(ctx != nullptr);
			assert(cubin_path != nullptr);
			CUresult e;

			e = cuCtxPushCurrent(ctx->context);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_load_module(): cuCtxPushCurrent() failed.",
					"x17ai_cuda_load_module(): cuCtxPushCurrent() failed with cuda error: ", e
				);
			}

			CUmodule module = nullptr;
			DiagnosticBuffer *diagnostic = nullptr;
			e = cuModuleLoad(&module, cubin_path);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				diagnostic = X17AI_DIAG(
					"x17ai_cuda_load_module(): cuModuleLoad() failed.",
					"x17ai_cuda_load_module(): cuModuleLoad() failed with cuda error: ", e
				);
			}

			CUcontext popped_ctx = nullptr;
			CUresult pop_err = cuCtxPopCurrent(&popped_ctx);
			if (pop_err != CUDA_SUCCESS) [[unlikely]] {
				if (module != nullptr) {
					[[maybe_unused]] CUresult ignored = cuModuleUnload(module);
				}
				if (diagnostic == nullptr) {
					diagnostic = X17AI_DIAG(
						"x17ai_cuda_load_module(): cuCtxPopCurrent() failed.",
						"x17ai_cuda_load_module(): cuCtxPopCurrent() failed with cuda error: ", pop_err
					);
				}
				return diagnostic;
			}
			if (diagnostic != nullptr) {
				return diagnostic;
			}
			if (popped_ctx != ctx->context) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_load_module(): popped unexpected CUDA context"
				);
			}
			if (module == nullptr) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_load_module(): cuModuleLoad() returned nullptr"
				);
			}

			assert_no_context("x17ai_cuda_load_module(): after cuCtxPopCurrent()");
			return reinterpret_cast<CudaModuleHandle *>(module);
		} X17AI_CATCH_ERRORS("x17ai_cuda_load_module()")
	}

	DiagnosticBuffer *x17ai_cuda_del_module(CudaModuleHandle *mod) noexcept {
		try {
			assert_no_context("x17ai_cuda_del_module(): before cuModuleUnload()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(mod != nullptr);
			CUmodule module = reinterpret_cast<CUmodule>(mod);
			CUresult e = cuModuleUnload(module);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_del_module(): cuModuleUnload() failed.",
					"x17ai_cuda_del_module(): cuModuleUnload() failed with cuda error: ", e
				);
			}
			assert_no_context("x17ai_cuda_del_module(): after cuModuleUnload()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_del_module()")
	}

	PtrResult<CudaKernelHandle> x17ai_cuda_get_kernel(
		CudaModuleHandle *mod,
		char const *name,
		usize smem_size
	) noexcept {
		try {
			assert_no_context("x17ai_cuda_get_kernel(): before cuModuleGetFunction()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(mod != nullptr);
			assert(name != nullptr);
			CUmodule module = reinterpret_cast<CUmodule>(mod);
			CUfunction kernel = nullptr;
			CUresult e = cuModuleGetFunction(&kernel, module, name);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_get_kernel(): cuModuleGetFunction() failed.",
					"x17ai_cuda_get_kernel(): cuModuleGetFunction() failed with cuda error: ", e
				);
			}
			if (kernel == nullptr) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_get_kernel(): cuModuleGetFunction() returned nullptr"
				);
			}

			if (!std::in_range<int>(smem_size)) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_get_kernel(): max dynamic shared memory size is out of range"
				);
			}
			e = cuFuncSetAttribute(
				kernel,
				CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
				int(unsigned(smem_size))
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_get_kernel(): cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) failed.",
					"x17ai_cuda_get_kernel(): cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) failed with cuda error: ", e
				);
			}

			e = cuFuncSetAttribute(
				kernel,
				CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
				100
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_get_kernel(): cuFuncSetAttribute(PREFERRED_SHARED_MEMORY_CARVEOUT) failed.",
					"x17ai_cuda_get_kernel(): cuFuncSetAttribute(PREFERRED_SHARED_MEMORY_CARVEOUT) failed with cuda error: ", e
				);
			}

			assert_no_context("x17ai_cuda_get_kernel(): after cuModuleGetFunction()");
			return reinterpret_cast<CudaKernelHandle *>(kernel);
		} X17AI_CATCH_ERRORS("x17ai_cuda_get_kernel()")
	}

	DiagnosticBuffer *x17ai_cuda_launch_kernel(
		CudaStreamHandle *stream, CudaKernelHandle *kernel,
		usize grid_x, usize grid_y, usize grid_z,
		usize block_x, usize block_y, usize block_z,
		usize shared_mem_bytes,
		void **args
	) noexcept {
		try {
			assert_no_context("x17ai_cuda_launch_kernel(): before cuLaunchKernel()");
			assert(cuda_initialized.load(std::memory_order_acquire));
			assert(stream != nullptr);
			assert(kernel != nullptr);

			if (
				(grid_x < 1 || !std::in_range<u32>(grid_x))
				|| (grid_y < 1 || !std::in_range<u32>(grid_y))
				|| (grid_z < 1 || !std::in_range<u32>(grid_z))
			) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_launch_kernel(): grid dimensions are out of range"
				);
			}
			if (
				(block_x < 1 || !std::in_range<u32>(block_x))
				|| (block_y < 1 || !std::in_range<u32>(block_y))
				|| (block_z < 1 || !std::in_range<u32>(block_z))
			) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_launch_kernel(): block dimensions are out of range"
				);
			}
			if (!std::in_range<u32>(shared_mem_bytes)) [[unlikely]] {
				return X17AI_STATIC_DIAG(
					"x17ai_cuda_launch_kernel(): shared memory size is out of range"
				);
			}

			CUstream cu_stream = reinterpret_cast<CUstream>(stream);
			CUfunction cu_kernel = reinterpret_cast<CUfunction>(kernel);
			CUresult e = cuLaunchKernel(
				cu_kernel,
				u32(grid_x), u32(grid_y), u32(grid_z),
				u32(block_x), u32(block_y), u32(block_z),
				u32(shared_mem_bytes),
				cu_stream,
				args,
				nullptr
			);
			if (e != CUDA_SUCCESS) [[unlikely]] {
				return X17AI_DIAG(
					"x17ai_cuda_launch_kernel(): cuLaunchKernel() failed.",
					"x17ai_cuda_launch_kernel(): cuLaunchKernel() failed with cuda error: ", e
				);
			}

			assert_no_context("x17ai_cuda_launch_kernel(): after cuLaunchKernel()");
			return nullptr;
		} X17AI_CATCH_ERRORS("x17ai_cuda_launch_kernel()")
	}
}
