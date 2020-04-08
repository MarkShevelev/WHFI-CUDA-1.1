#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace diffusion{ namespace device {
	template <unsigned TILE_SIZE, typename T>
	__global__ void  forward_step_matrix_calculation_kernel(T r, T const *x, T const *dfc, T *a, T *b, T *c, T *d, unsigned row_size, unsigned row_count) {
		unsigned row_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
		unsigned elm_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
		unsigned idx = row_idx + elm_idx * row_count;

		if (0 == elm_idx || row_size - 1 == elm_idx ) return;

		a[idx] = -0.5 * r * dfc[idx - row_count];
		b[idx] = T(1) + 0.5 * r * (dfc[idx - row_count] + dfc[idx]);
		c[idx] = -0.5 * r * dfc[idx];
		d[idx] = x[idx] + 0.5 * r * (fma(dfc[idx], x[idx + row_count], -dfc[idx] * x[idx]) + fma(dfc[idx - row_count], x[idx - row_count], -dfc[idx - row_count] * x[idx]));
	}
}/*device*/ }/*diffusion*/ }/*iki*/