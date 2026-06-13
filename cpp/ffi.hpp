#pragma once

#include <cuda.h>
#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace {

	struct FfiSpan {
		char *ptr;
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

		inline bool write() noexcept {
			return true;
		}

		inline bool write(std::string_view str) noexcept {
			size_t len = str.size();
			std::span buf = extend(len);
			if (buf.size() != len) [[unlikely]] {
				return false;
			}
			std::copy(str.begin(), str.end(), buf.data());
			return true;
		}

		inline bool write(CUresult e) noexcept {
			const char *err_str = nullptr;
			if (cuGetErrorString(e, &err_str) == CUDA_SUCCESS) [[likely]] {
				return write(err_str);
			} else {
				return write("Unknown CUDA error code");
			}
		}

		template<typename A, typename B, typename... CS>
		inline bool write(A const &a, B const &b, CS const &... cs) noexcept {
			if (!write(a)) {
				return false;
			}
			return write(b, cs...);
		}

		inline void clear() noexcept {
			vmt->clear(instance);
		}

		inline void set_len(size_t new_len) noexcept {
			vmt->set_len(instance, new_len);
		}
	};

	struct DiagnosticBuffer {
		size_t is_allocated;
		std::string_view message;

		constexpr DiagnosticBuffer(std::string_view msg):
			is_allocated(0),
			message(msg)
		{}
	};

	struct AllocatedDiagnosticBuffer: DiagnosticBuffer {
		std::string buffer;

		AllocatedDiagnosticBuffer(std::string_view fallback_msg):
			DiagnosticBuffer(fallback_msg),
			buffer()
		{
			is_allocated = 1;
		}

		bool write() noexcept {
			return true;
		}

		bool write(std::string_view str) noexcept {
			bool ok;
			try {
				buffer.append(str);
				ok = true;
			} catch (...) {
				ok = false;
			}
			message = buffer;
			return ok;
		}

		bool write(CUresult e) noexcept {
			const char *err_str = nullptr;
			if (cuGetErrorString(e, &err_str) == CUDA_SUCCESS) [[likely]] {
				return write(err_str);
			} else {
				return write("Unknown CUDA error code");
			}
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
	DiagnosticBuffer *new_diag(DiagnosticBuffer *fallback, TS const &... ts) noexcept {
		try {
			auto result = std::make_unique<AllocatedDiagnosticBuffer>(fallback->message);
			result->write(ts...);
			return result.release();
		} catch (...) {
			return fallback;
		}
	}

	#define X17AI_DIAG(FALLBACK, ...) \
		([&]() noexcept -> DiagnosticBuffer * { \
			static DiagnosticBuffer fallback_diag{FALLBACK}; \
			return new_diag(&fallback_diag, __VA_ARGS__); \
		}())

	#define X17AI_STATIC_DIAG(FALLBACK) \
		([]() noexcept -> DiagnosticBuffer * { \
			static DiagnosticBuffer fallback_diag{FALLBACK}; \
			return &fallback_diag; \
		}())

	template<typename T>
	struct PtrResult {
		T *value;
		DiagnosticBuffer *diagnostic;

		PtrResult(T *val):
			value(val),
			diagnostic(nullptr)
		{}

		PtrResult(DiagnosticBuffer *diag):
			value(nullptr),
			diagnostic(diag)
		{}
	};

	struct NonNullContextError {
		char const *where;

		explicit NonNullContextError(char const *where):
			where(where)
		{}
	};

	struct ContextQueryError {
		char const *where;
		CUresult error;

		ContextQueryError(char const *where, CUresult error):
			where(where),
			error(error)
		{}
	};

	inline void assert_no_context(char const *where) {
		#ifndef NDEBUG
		CUcontext pctx = nullptr;
		CUresult err = cuCtxGetCurrent(&pctx);
		if (err != CUDA_SUCCESS) {
			throw ContextQueryError(where, err);
		}
		if (pctx != nullptr) {
			throw NonNullContextError(where);
		}
		#endif
	}
}

#define X17AI_CATCH_ERRORS(FUNCTION_NAME) \
	catch (ContextQueryError const &e) { \
		return X17AI_DIAG( \
			FUNCTION_NAME ": CUDA context query failed.", \
			FUNCTION_NAME ": CUDA context query failed at ", e.where, \
			" with cuda error: ", e.error \
		); \
	} catch (NonNullContextError const &e) { \
		return X17AI_DIAG( \
			FUNCTION_NAME ": unexpected CUDA context.", \
			FUNCTION_NAME ": unexpected CUDA context at ", e.where \
		); \
	} catch (std::exception const &e) { \
		return X17AI_DIAG( \
			FUNCTION_NAME ": exception thrown.", \
			FUNCTION_NAME ": exception thrown: ", e.what() \
		); \
	} catch (...) { \
		return X17AI_STATIC_DIAG(FUNCTION_NAME ": unknown exception thrown"); \
	}
