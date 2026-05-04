#pragma once

#include "base_matmul.cuh"

#pragma nv_diag_suppress 186

template<
	const usize D_IN,
	const usize D_OUT,
	const usize _FAN_IN,
	const usize _CYCLE
>
struct SparseMatMul:
	BaseMatMul<
		D_IN,
		D_OUT,
		_FAN_IN,
		_CYCLE
	>
{
};
