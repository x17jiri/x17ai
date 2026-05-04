#pragma once

#include "base_matmul.cuh"

#pragma nv_diag_suppress 186

template<
	const usize D_IN,
	const usize D_OUT
>
struct DenseMatMul:
	BaseMatMul<
		D_IN,
		D_OUT
	>
{
};
