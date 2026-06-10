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

	static_assert(sizeof(usize) == 8, "usize: invalid size");
	static_assert(alignof(usize) == 8, "usize: invalid alignment");
	static_assert(sizeof(void *) == 8, "pointer: invalid size");
	static_assert(alignof(void *) == 8, "pointer: invalid alignment");

	struct KernelOutput {
		usize size[3];
		usize stride_bytes[3];
		usize offset_bytes;
		void *buf;
	};

	static_assert(sizeof(KernelOutput) == 64, "KernelOutput: invalid size");
	static_assert(alignof(KernelOutput) == 8, "KernelOutput: invalid alignment");

	struct KernelArg {
		usize stride_bytes[3];
		usize offset_bytes;
		void *buf;
	};

	static_assert(sizeof(KernelArg) == 40, "KernelArg: invalid size");
	static_assert(alignof(KernelArg) == 8, "KernelArg: invalid alignment");

	template<typename T, const usize N>
	struct Array {
		T items[N];
	};

	using InternalDtype = {{internal_dtype}};
	using OutputDtype = {{out_dtype}};

	{% let (elem_args, reduce_args) = tensor_args.split_at(tensor_args.len() - reduce_args_count) -%}

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

		inline __device__ void append(T value) {
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

	template<typename T>
	struct Max {
		T max;

		inline __device__ Max(T value): max(value) {}

		inline __device__ void append(T value) {
			max = fmax(max, value);
		}

		inline __device__ T result() const {
			return max;
		}
	};

	template<typename T>
	inline __device__ T pairwise_max(T a, T b) {
		return fmax(a, b);
	}

	__device__ __forceinline__ double inf() {
		return __longlong_as_double(0x7ff0000000000000ULL);
	}
}

extern "C" __global__ void x17ai_kernel(
	KernelOutput o_arg
	{% if tensor_args.len() > 0 %}, Array<KernelArg, E_CNT + R_CNT> t_args{% endif %}
	{% if scalar_args_count > 0 %}, Array<f64, S_CNT> scalars{% endif %}
) {
	usize w = o_arg.size[2];
	usize h = o_arg.size[1];
	usize x = threadIdx.x;
	usize y = blockIdx.x;
	if (y >= h) {
		return;
	}

	// Calculate reduce argument pointers
	{%- for i in 0..reduce_args.len() %}
	KernelArg &r_arg{{i}} = t_args.items[E_CNT + {{i}}];
	R{{i}}Dtype *r{{i}}_ptr = reinterpret_cast<R{{i}}Dtype *>(
		reinterpret_cast<char *>(r_arg{{i}}.buf)
		+ r_arg{{i}}.offset_bytes
		+ y * r_arg{{i}}.stride_bytes[1]
		+ x * r_arg{{i}}.stride_bytes[2]
	);
	{%- endfor %}

	// Get scalar arguments
	{%- for s in 0..scalar_args_count %}
	InternalDtype s{{s}} = static_cast<InternalDtype>(scalars.items[{{s}}]);
	{%- endfor %}

	InternalDtype reduce_val = {{identity}};
	if (x < w) {
		{%- for i in 0..reduce_args.len() %}
		InternalDtype r{{i}} = static_cast<InternalDtype>(*r{{i}}_ptr);
		{%- endfor %}
		reduce_val = {{pre_reduce_expr}};
	}
	if (blockDim.x < w) {
		{%- for i in 0..reduce_args.len() %}
		usize r{{i}}_stride = r_arg{{i}}.stride_bytes[2] * blockDim.x;
		{%- endfor %}
		{{loop_reduce}}<InternalDtype> kahan_sum(reduce_val);
		for (usize i = blockDim.x; i < w; i += blockDim.x) {
			{%- for i in 0..reduce_args.len() %}
			r{{i}}_ptr = reinterpret_cast<R{{i}}Dtype *>(
				reinterpret_cast<char *>(r{{i}}_ptr)
				+ r{{i}}_stride
			);
			{%- endfor %}
			if (i + x < w) {
				{%- for i in 0..reduce_args.len() %}
				InternalDtype r{{i}} = static_cast<InternalDtype>(*r{{i}}_ptr);
				{%- endfor %}
				kahan_sum.append({{pre_reduce_expr}});
			}
		}
		reduce_val = kahan_sum.result();
	}

	// First warp-level reduction
	using WarpMask = decltype(__activemask());
	WarpMask warp_mask;
	if constexpr (8*sizeof(WarpMask) > WARP_SIZE) {
		warp_mask = (WarpMask(1) << WARP_SIZE) - WarpMask(1);
	} else {
		warp_mask = ~WarpMask(0);
	}
	// assert(warp_mask == __activemask());

	#pragma unroll
	for (int t = WARP_SIZE / 2; t > 0; t /= 2) {
		reduce_val = {{pairwise_reduce}}(reduce_val, __shfl_down_sync(warp_mask, reduce_val, t));
	}

	__shared__ InternalDtype shared[WARP_SIZE];
	if (x % WARP_SIZE == 0) {
		shared[x / WARP_SIZE] = reduce_val;
	}
	__syncthreads();

	// Second warp-level reduction
	if (x < WARP_SIZE) {
		InternalDtype new_val = shared[x];
		if (x >= (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
			new_val = {{identity}};
		}
		#pragma unroll
		for (int t = WARP_SIZE / 2; t > 0; t /= 2) {
			new_val = {{pairwise_reduce}}(new_val, __shfl_down_sync(warp_mask, new_val, t));
		}
		reduce_val = new_val;
	}

	if (o_arg.stride_bytes[2] == 0) {
		if (x == 0) {
			InternalDtype post_reduce_common = {{post_reduce_common}};

			// Calculate elementwise argument pointers
			{%- for i in 0..elem_args.len() %}
			KernelArg &e_arg{{i}} = t_args.items[{{i}}];
			E{{i}}Dtype *e{{i}}_ptr = reinterpret_cast<E{{i}}Dtype *>(
				reinterpret_cast<char *>(e_arg{{i}}.buf)
				+ e_arg{{i}}.offset_bytes
				+ y * e_arg{{i}}.stride_bytes[1]
				// + x * e_arg{{i}}.stride_bytes[2] // x is 0 here
			);
			InternalDtype e{{i}} = static_cast<InternalDtype>(*e{{i}}_ptr);
			{%- endfor %}

			// Calculate output pointer
			OutputDtype *o = reinterpret_cast<OutputDtype *>(
				reinterpret_cast<char *>(o_arg.buf)
				+ o_arg.offset_bytes
				+ y * o_arg.stride_bytes[1]
				// + x * o_arg.stride_bytes[2] // x is 0 here
			);

			*o = static_cast<OutputDtype>(
				{{post_reduce_expr}}
			);
		}
	} else {
		if (x == 0) {
			shared[0] = {{post_reduce_common}};
		}
		__syncthreads();
		InternalDtype post_reduce_common = shared[0];

		// Calculate elementwise argument pointers
		{%- for i in 0..elem_args.len() %}
		KernelArg &e_arg{{i}} = t_args.items[{{i}}];
		E{{i}}Dtype *e{{i}}_ptr = reinterpret_cast<E{{i}}Dtype *>(
			reinterpret_cast<char *>(e_arg{{i}}.buf)
			+ e_arg{{i}}.offset_bytes
			+ y * e_arg{{i}}.stride_bytes[1]
			+ x * e_arg{{i}}.stride_bytes[2]
		);
		{%- endfor %}

		// Calculate output pointer
		OutputDtype *o = reinterpret_cast<OutputDtype *>(
			reinterpret_cast<char *>(o_arg.buf)
			+ o_arg.offset_bytes
			+ y * o_arg.stride_bytes[1]
			+ x * o_arg.stride_bytes[2]
		);

		if (x < w) {
			{%- for i in 0..elem_args.len() %}
			InternalDtype e{{i}} = static_cast<InternalDtype>(*e{{i}}_ptr);
			{%- endfor %}
			*o = static_cast<OutputDtype>(
				{{post_reduce_expr}}
			);
		}
		if (blockDim.x < w) {
			{%- for i in 0..elem_args.len() %}
			usize e{{i}}_stride = e_arg{{i}}.stride_bytes[2] * blockDim.x;
			{%- endfor %}
			usize o_stride = o_arg.stride_bytes[2] * blockDim.x;
			for (usize i = blockDim.x; i < w; i += blockDim.x) {
				if (i + x < w) {
					{%- for i in 0..elem_args.len() %}
					e{{i}}_ptr = reinterpret_cast<E{{i}}Dtype *>(
						reinterpret_cast<char *>(e{{i}}_ptr)
						+ e{{i}}_stride
					);
					InternalDtype e{{i}} = static_cast<InternalDtype>(*e{{i}}_ptr);
					{%- endfor %}
					o = reinterpret_cast<OutputDtype *>(
						reinterpret_cast<char *>(o)
						+ o_stride
					);
					*o = static_cast<OutputDtype>(
						{{post_reduce_expr}}
					);
				}
			}
		}
	}
}
