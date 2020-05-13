#pragma once

#include "diagonal_discretization.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace diffusion { namespace device {
	template <unsigned TILE_SIZE, typename T>
	__global__ void  correction_forward_step_matrix_calculation_kernel(unsigned row_count, unsigned row_size, T *a, T *b, T *c, T *d, T const *x_prev, T const *x_next, T const *along_dfc, T along_r) {
		unsigned row_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
		unsigned elm_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
		unsigned idx = row_idx + elm_idx * row_count;

		if (0 == row_idx || row_count - 1 == row_idx || 0 == elm_idx || row_size - 1 == elm_idx) return;

		a[idx] = -0.5 * along_r * along_dfc[idx - row_count];
		b[idx] = T(1) + 0.5 * along_r * (along_dfc[idx - row_count] + along_dfc[idx]);
		c[idx] = -0.5 * along_r * along_dfc[idx];
		d[idx] = x_next[idx] 
			- 0.5 * along_r * diagonal_discretization(x_prev[idx - row_count], x_prev[idx], x_prev[idx + row_count], along_dfc[idx - row_count], along_dfc[idx]);
	}
}/*device*/ }/*diffusion*/ }/*iki*/