//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

namespace {

	using usize = size_t;

	using u64 = unsigned long long;

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
		usize reduction_stride_bytes;
	};

	static_assert(sizeof(KernelOutput) == 64, "KernelOutput must be 64 bytes");
	static_assert(alignof(KernelOutput) == 8, "KernelOutput must be 8 bytes aligned");

	struct KernelReduceArg {
		usize stride_bytes[3];
		void *buf;
		usize offset_bytes;
	};

	static_assert(sizeof(KernelReduceArg) == 40, "KernelReduceArg must be 40 bytes");
	static_assert(alignof(KernelReduceArg) == 8, "KernelReduceArg must be 8 bytes aligned");

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

	{% for r in reduce_args -%}
	using R{{loop.index0}}Dtype = {{r.dtype}};
	{% endfor %}

	constexpr usize WARP_SIZE = {{warp_size}};
	constexpr usize R_CNT = {{reduce_args.len()}};
	constexpr usize E_CNT = {{elem_args.len()}};
	constexpr usize S_CNT = {{scalar_args_count}};

	inline __device__ bool is_power_of_two(usize x) {
		return (x & (x - 1)) == 0;
	}

	inline __device__ usize trailing_zeros(usize x) {
		return __ffsll(x) - 1;
	}

	template<typename T>
	struct KahanSum {
		T sum;
		T c;

		inline __device__ KahanSum(T value): sum(value), c(0) {}

		inline __device__ void add(T value) {
			T y = value - c;
			T t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}

		inline __device__ T result() const {
			return sum;
		}
	};

	template<typename T>
	inline __device__ T pairwise_sum(T a, T b) {
		return a + b;
	}

}

extern "C" __global__ void x17ai_kernel(
	KernelOutput o_arg
	, Array<KernelReduceArg, R_CNT> r_args
	{% if elem_args.len() > 0 %}, Array<KernelElemArg, E_CNT> e_args{% endif %}
	{% if scalar_args_count > 0 %}, Array<f64, S_CNT> scalars{% endif %}
) {
	// `x`, `y` are the indices along the output dimensions
	usize idx = usize(blockIdx.x);
	usize w = o_arg.size[1];
	usize h = o_arg.size[0];
	usize x, y;
	if (is_power_of_two(w)) {
		x = idx & (w - 1);
		y = idx >> trailing_zeros(w);
	} else {
		x = idx % w;
		y = idx / w;
	}
	if (x >= w || y >= h) {
		return;
	}

	// `z` is the index along the reduction dimension
	usize z = usize(threadIdx.x);

	// Calculate output pointer
	OutputDtype *o = reinterpret_cast<OutputDtype *>(
		reinterpret_cast<char *>(o_arg.buf)
		+ o_arg.offset_bytes
		+ y * o_arg.stride_bytes[0]
		+ x * o_arg.stride_bytes[1]
	);

	// Calculate reduce argument pointers
	{%- for i in 0..reduce_args.len() %}
	KernelReduceArg &r_arg{{i}} = r_args.items[{{i}}];
	R{{i}}Dtype *r{{i}}_ptr = reinterpret_cast<R{{i}}Dtype *>(
		reinterpret_cast<char *>(r_arg{{i}}.buf)
		+ r_arg{{i}}.offset_bytes
		+ y * r_arg{{i}}.stride_bytes[0]
		+ x * r_arg{{i}}.stride_bytes[1]
		+ z * r_arg{{i}}.stride_bytes[2]
	);
	{%- endfor %}

	// Get scalar arguments
	{%- for s in 0..scalar_args_count %}
	InternalDtype s{{s}} = static_cast<InternalDtype>(scalars.items[{{s}}]);
	{%- endfor %}

	InternalDtype val = {{zero}};
	if (z < o_arg.reduction_size) {
		{%- for i in 0..reduce_args.len() %}
		InternalDtype r{{i}} = static_cast<InternalDtype>(*r{{i}}_ptr);
		{%- endfor %}
		val = {{pre_reduce_expr}};
	}
	if (blockDim.x < o_arg.reduction_size) {
		{%- for i in 0..reduce_args.len() %}
		usize r{{i}}_stride = r_arg{{i}}.stride_bytes[2] * blockDim.x;
		{%- endfor %}
		KahanSum<InternalDtype> kahan_sum(val);
		for (usize i = blockDim.x; i < o_arg.reduction_size; i += blockDim.x) {
			{%- for i in 0..reduce_args.len() %}
			r{{i}}_ptr = reinterpret_cast<R{{i}}Dtype *>(
				reinterpret_cast<char *>(r{{i}}_ptr)
				+ r{{i}}_stride
			);
			{%- endfor %}
			{%- for i in 0..reduce_args.len() %}
			InternalDtype r{{i}} = static_cast<InternalDtype>(*r{{i}}_ptr);
			{%- endfor %}
			if (i + z < o_arg.reduction_size) {
				kahan_sum.add({{pre_reduce_expr}});
			}
		}
		val = kahan_sum.result();
	}
	unsigned mask = __activemask();
	for (int t = WARP_SIZE / 2; t > 0; t /= 2) {
		val = pairwise_sum(val, __shfl_down_sync(0xFFFFFFFF, val, t));
	}

	static __shared__ InternalDtype shared[WARP_SIZE];
	if (z % WARP_SIZE == 0) {
		shared[z / WARP_SIZE] = val;
	}
	__syncthreads();

	if (z < WARP_SIZE) {
		InternalDtype new_val = {{zero}};
		if (z < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
			new_val = shared[z];
		}
		for (int t = WARP_SIZE / 2; t > 0; t /= 2) {
			new_val = pairwise_sum(new_val, __shfl_down_sync(0xFFFFFFFF, new_val, t));
		}
		val = new_val;
	}

	if (o_arg.reduction_stride_bytes == 0) {
		if (z == 0) {
			val = {{post_reduce_expr}};
			*o = static_cast<OutputDtype>(val);
		}
	} else {
		if (z == 0) {
			shared[0] = val;
		}
		__syncthreads();
		val = shared[0];

		// TODO
		/*
		// Calculate elementwise argument pointers
		{%- for i in 0..elem_args.len() %}
		KernelElemArg &e_arg{{i}} = e_args.items[{{i}}];
		InternalDtype e{{i}} = static_cast<InternalDtype>(
			*reinterpret_cast<E{{i}}Dtype *>(
				reinterpret_cast<char *>(e_arg{{i}}.buf)
				+ e_arg{{i}}.offset_bytes
				+ y * e_arg{{i}}.stride_bytes[0]
				+ x * e_arg{{i}}.stride_bytes[1]
			)
		);
		{%- endfor %}

		// Store the result
		*o = static_cast<OutputDtype>(value);
		*/
	}
}
