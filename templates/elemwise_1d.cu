//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

using usize = size_t;

using f32 = float;
using f64 = double;

using F32 = f32;
using F64 = f64;

static_assert(sizeof(usize) == 8, "usize must be 8 bytes");
static_assert(alignof(usize) == 8, "usize must be 8 bytes aligned");
static_assert(sizeof(void *) == 8, "pointer must be 8 bytes");
static_assert(alignof(void *) == 8, "pointer must be 8 bytes aligned");

struct KernelOutput {
	usize size[2];
	usize stride_bytes[2];
	void *buf;
	usize offset_bytes;
	usize reduction_size;
};

static_assert(sizeof(KernelOutput) == 56, "KernelOutput must be 56 bytes");
static_assert(alignof(KernelOutput) == 8, "KernelOutput must be 8 bytes aligned");

struct KernelElemArg {
	usize stride_bytes[2];
	void *buf;
	usize offset_bytes;
};

static_assert(sizeof(KernelElemArg) == 32, "KernelElemArg must be 32 bytes");
static_assert(alignof(KernelElemArg) == 8, "KernelElemArg must be 8 bytes aligned");

template<typename T, const usize N>
struct Array {
	T items[N];
};

using InternalDtype = {{internal_dtype}};
using OutputDtype = {{out_dtype}};

{% for e in elem_args -%}
using E{{loop.index0}}Dtype = {{e.dtype}};
{% endfor %}

constexpr usize E_CNT = {{elem_args.len()}};
constexpr usize S_CNT = {{scalar_args_count}};

extern "C" __global__ void x17ai_kernel(
	KernelOutput o_arg
	{% if elem_args.len() > 0 %}, Array<KernelElemArg, E_CNT> e_args{% endif %}
	{% if scalar_args_count > 0 %}, Array<f64, S_CNT> scalars{% endif %}
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= o_arg.size[1]) {
		return;
	}

	// Calculate output pointer
	OutputDtype *o = reinterpret_cast<OutputDtype *>(
		reinterpret_cast<char *>(o_arg.buf)
		+ o_arg.offset_bytes
		+ idx * o_arg.stride_bytes[1]
	);

	// Calculate elementwise argument pointers
	{%- for i in 0..elem_args.len() %}
	KernelElemArg e_arg{{i}} = e_args.items[{{i}}];
	InternalDtype e{{i}} = static_cast<InternalDtype>(
		*reinterpret_cast<E{{i}}Dtype *>(
			reinterpret_cast<char *>(e_arg{{i}}.buf)
			+ e_arg{{i}}.offset_bytes
			+ idx * e_arg{{i}}.stride_bytes[1]
		)
	);
	{%- endfor %}

	// Get scalar arguments
	{%- for s in 0..scalar_args_count %}
	InternalDtype s{{s}} = static_cast<InternalDtype>(scalars.items[{{s}}]);
	{%- endfor %}

	// Evaluate expression
	InternalDtype value = {{expr}};

	// Store the result
	*o = static_cast<OutputDtype>(value);
}
