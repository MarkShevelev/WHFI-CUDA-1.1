#pragma once

#include "thomson_sweep.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace math { namespace device { 
	template <typename T>
	__global__ void thomson_sweep_kernel(T *a, T *b, T *c, T *d, T *x, unsigned row_size, unsigned row_count) {
		unsigned row_id = threadIdx.x + blockIdx.x * blockDim.x;
		if (0 == row_id || row_count - 1 == row_id) return;
		thomson_sweep(a + row_id, b + row_id, c + row_id, d + row_id, x + row_id, row_size, row_count);
	}
}/*device*/ }/*math*/ }/*iki*/