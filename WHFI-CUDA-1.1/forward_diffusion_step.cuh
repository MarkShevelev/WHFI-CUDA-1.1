#pragma once

#include "thomson_sweep_kernel.cuh"
#include "forward_step_matrix_calculation_kernel.cuh"

#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device {
	template <typename T, unsigned TILE_SIZE, unsigned THOMSON_THREADS>
	void forward_diffusion_step(T *a, T *b, T *c, T *d, T const *x_curr, T *x_next, T r_along, T const *dfc_along, T r_perp, T const *dfc_perp, unsigned row_size, unsigned row_count) {

		{
			dim3 blocks(row_count / TILE_SIZE, row_size / TILE_SIZE), treads(TILE_SIZE, TILE_SIZE);
			forward_step_matrix_calculation_kernel<TILE_SIZE> << <blocks, threads >> > (a, b, c, d, x_curr, r_along, dfc_along, r_perp, dfc_perp, row_size, row_count);
		}

		{
			unsigned matrix_offset = row_count + 1;
			unsigned threads = THOMSON_THREADS, blocks = row_count / threads;
			math::device::thomson_sweep_kernel << <blocks, threads >> > (a + offset, b + offset, c + offset, d + offset, x_next + offset, row_size - 2, row_count);
		}
	}
}/*device*/ }/*diffusion*/ }/*iki*/