#pragma once

#include <cstddef>
#include <cstdint>

using std::size_t;

struct KernelElemArg {
	size_t stride_bytes[2];
	size_t offset_bytes;
	uint8_t const *device_data;
};

template<typename T>
struct TypedKernelElemArg {
	KernelElemArg const &arg;

	TypedKernelElemArg(KernelElemArg const &arg): arg(arg) {}

	inline T const &operator[](size_t i) const {
		uint8_t const *ptr = device_data + offset_bytes + (i * stride_bytes[0]);
		return *reinterpret_cast<T const *>(ptr);
	}
};

struct KernelReduceArg {
	size_t reduction_size;
	size_t stride_bytes[2];
	size_t offset;
	uint8_t *device_data;
};

template<size_t N>
struct KernelElemArgs {
	KernelElemArg args[N];

	inline KernelElemArg const &operator[](size_t i) const {
		return args[i];
	}
};

template<size_t N>
struct KernelReduceArgs {
	KernelReduceArg args[N];
};

template<size_t N>
struct KernelConstArgs {
	double args[N];
};

struct KernelOutput {
	size_t size[2];
	size_t stride_bytes[2];
	size_t offset_bytes;
	uint8_t *device_data;
};

template<typename T>
struct TypedKernelOutput {
	KernelOutput const &out;

	TypedKernelOutput(KernelOutput const &out): out(out) {}

	inline T &operator[](size_t i) const {
		uint8_t *ptr = device_data + offset_bytes + (i * stride_bytes[0]);
		return *reinterpret_cast<T *>(ptr);
	}
};

extern "C" __global__ void cuda_kernel( //
	KernelOutput out_arg,
	KernelElemArgs<2> elem_args
) {
	auto out = TypedKernelOutput<float>(out_arg);
	auto a = TypedKernelElemArg<float>(elem_args[0]);
	auto b = TypedKernelElemArg<float>(elem_args[1]);

	// TODO: size[0] not used
	size_t const size = out_arg.size[1];
	// TODO: should the type be size_t or int?
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
		out[i] = a[i] * b[i];
	}
}
