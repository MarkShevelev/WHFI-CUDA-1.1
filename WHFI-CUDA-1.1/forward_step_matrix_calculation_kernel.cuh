#pragma once

#include "diagonal_discretization.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace diffusion{ namespace device {
	template <unsigned TILE_SIZE, typename T>
	__global__ void  forward_step_matrix_calculation_kernel(unsigned row_count, unsigned row_size, T *a, T *b, T *c, T *d, T const *x_curr, T const *along_dfc, T along_r, T const *perp_dfc, T perp_r) {
		unsigned row_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
		unsigned elm_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
		unsigned idx = row_idx + elm_idx * row_count;

		if (0 == row_idx || row_count - 1 == row_idx || 0 == elm_idx || row_size - 1 == elm_idx) return;

		a[idx] = -0.5 * along_r * along_dfc[idx - row_count];
		b[idx] = T(1) + 0.5 * along_r * (along_dfc[idx - row_count] + along_dfc[idx]);
		c[idx] = -0.5 * along_r * along_dfc[idx];
		d[idx] = x_curr[idx] 
			+ 0.5 * along_r * diagonal_discretization(x_curr[idx - row_count], x_curr[idx], x_curr[idx + row_count], along_dfc[idx - row_count], along_dfc[idx])
			+ perp_r * diagonal_discretization(x_curr[idx - 1], x_curr[idx], x_curr[idx + 1], perp_dfc[elm_idx + (row_idx - 1) * row_size], perp_dfc[elm_idx + row_idx * row_size]);
	}
}/*device*/ }/*diffusion*/ }/*iki*/