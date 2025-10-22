                                                                  using usize = size_t;

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

using Internal = {{INTERNAL_TYPE}};
using O = {{O_TYPE}};

{% for e in ES %}
using E{{loop.index0}} = {{e.type}};
{% endfor %}

constexpr usize E = {{ES|length}};

extern "C" __global__ void x17ai_kernel(KernelOutput o_arg, Array<KernelElemArg, E> e_args) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= o_arg.size[1]) {
		return;
	}

	O *o = reinterpret_cast<O *>(
		reinterpret_cast<char *>(o_arg.buf)
		+ o_arg.offset_bytes
		+ idx * o_arg.stride_bytes[1]
	);

	{% for e in ES %}
	Internal e{{loop.index0}} = Internal(
		*reinterpret_cast<E{{loop.index0}} *>(
			reinterpret_cast<char *>(e_args.items[0].buf)
			+ e_args.items[{{loop.index0}}].offset_bytes
			+ idx * e_args.items[{{loop.index0}}].stride_bytes[1]
		)
	);
	{% endfor %}

	*o = O({{EXPR}});
}
