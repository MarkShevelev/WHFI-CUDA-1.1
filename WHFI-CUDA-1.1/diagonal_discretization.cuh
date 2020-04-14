#pragma once

#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device {
	template <typename T>
	inline
	__device__ T diagonal_discretization(T const *x, T const *dfc, unsigned idx, unsigned stride  = 1) {
		return fma(dfc[idx], x[idx + stride], -dfc[idx] * x[idx]) + fma(dfc[idx - stride], x[idx - stride], -dfc[idx - stride] * x[idx]);
	}
}/*device*/ }/*diffusion*/ }/*iki*/