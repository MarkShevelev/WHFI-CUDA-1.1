#pragma once

#include "thomson_sweep_kernel.cuh"
#include "forward_step_matrix_calculation_kernel.cuh"

#include <cuda_runtime.h>
#include <algorithm>

namespace iki { namespace diffusion {
	template <unsigned TILE_SIZE, unsigned THREAD_COUNT, typename T>
	class OneDimensionalMultithreadDiffusion final {
	public:
		OneDimensionalMultithreadDiffusion(T r, unsigned row_size, unsigned row_count, T *x_next, T *x_prev, T *dfc, T *a, T *b, T *c, T *d): r(r), row_size(row_size), row_count(row_count), x_next(x_next), x_prev(x_prev), dfc(dfc), a(a), b(b), c(c), d(d) { }

		OneDimensionalMultithreadDiffusion& step() {
			dim3 blockDim(TILE_SIZE, TILE_SIZE), gridDim(row_size / TILE_SIZE, row_count / TILE_SIZE);
			device::forward_step_matrix_calculation_kernel<TILE_SIZE><<<step_gridDim,step_blockDim>>>(r, x_prev, dfc, a, b, c, d, row_size, row_count);
			cudaDeviceSynchronize();

			math::device::thomson_sweep_kernel<<<row_count/THREAD_COUNT, THREAD_COUNT>>>(a + row_count, b + row_count, c + row_count, d + row_count, x_next + row_count, row_size-2, row_count);
			cudaDeviceSynchronize();

			std::swap(x_prev, x_next);
			return *this;
		}

	public:
		T const r;
		unsigned const row_size, row_count;
		T *x_next, *x_prev, *dfc, *a, *b, *c, *d; //device memory pointers
	};
} /*diffusion*/ } /*iki*/