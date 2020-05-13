#pragma once

#include "thomson_sweep_kernel.cuh"
#include "cycle_grid_transpose.cuh"
#include "forward_step_matrix_calculation_kernel.cuh"
#include "correction_forward_step_matrix_calculation_kernel.cuh"

#include <cuda_runtime.h>
#include <algorithm>

namespace iki {	namespace diffusion {
	template <unsigned TILE_SIZE, unsigned THREAD_COUNT, typename T>
	class TwoDimensionalMultithreadDiffusion final {
	public:
		TwoDimensionalMultithreadDiffusion(unsigned row_count, unsigned row_size, T *a, T *b, T *c, T *d, T *x_prev, T *x_next, T *x_tmp, T *along_dfc, T along_r, T *perp_dfc, T perp_r ): row_count(row_count), row_size(row_size), a(a), b(b), c(c), d(d), x_prev(x_prev), x_next(x_next), x_tmp(x_tmp), along_dfc(along_dfc), perp_dfc(perp_dfc), along_r(along_r), perp_r(perp_r) { }

		TwoDimensionalMultithreadDiffusion& step() {
			{
				dim3 blockDim(TILE_SIZE, TILE_SIZE), gridDim(row_count / TILE_SIZE, row_size / TILE_SIZE);
				device::forward_step_matrix_calculation_kernel<TILE_SIZE><<<gridDim, blockDim>>> (row_count, row_size, a, b, c, d, x_prev, along_dfc, along_r, perp_dfc, perp_r);
				cudaDeviceSynchronize();

				math::device::thomson_sweep_kernel<<<row_count / THREAD_COUNT, THREAD_COUNT>>>(a + row_count, b + row_count, c + row_count, d + row_count, x_next + row_count, row_size - 2, row_count);
				cudaDeviceSynchronize();

				math::device::cycle_grids_transpose<TILE_SIZE, 8u>(x_prev, x_next, x_tmp, row_count, row_size);
				std::swap(x_next, x_tmp);
				std::swap(x_prev, x_tmp);
				std::swap(row_count, row_size);
			}

			{
				dim3 blockDim(TILE_SIZE, TILE_SIZE), gridDim(row_count / TILE_SIZE, row_size / TILE_SIZE);
				device::correction_forward_step_matrix_calculation_kernel<TILE_SIZE><<<gridDim, blockDim>>> (row_count, row_size, a, b, c, d, x_prev, x_next, perp_dfc, perp_r);
				cudaDeviceSynchronize();

				math::device::thomson_sweep_kernel<<<row_count / THREAD_COUNT, THREAD_COUNT>>> (a + row_count, b + row_count, c + row_count, d + row_count, x_next + row_count, row_size - 2, row_count);
				cudaDeviceSynchronize();

				math::device::cycle_grids_transpose<TILE_SIZE, 8u>(x_prev, x_next, x_tmp, row_count, row_size);
				std::swap(x_next, x_tmp);
				std::swap(x_prev, x_tmp);
				std::swap(row_count, row_size);
			}

			std::swap(x_prev, x_next);
			return *this;
		}

	public:
		unsigned row_count, row_size;
		T *a, *b, *c, *d, *x_prev, *x_next, *x_tmp, *along_dfc, *perp_dfc; //device memory pointers
		T const along_r, perp_r;
	};
} /*diffusion*/ } /*iki*/