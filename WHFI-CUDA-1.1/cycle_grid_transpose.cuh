#pragma once

#include "transpose.cuh"
#include <cuda_runtime.h>

namespace iki { namespace math { namespace device {
	template <unsigned TILE_SIZE, unsigned BLOCK_ROWS, typename T>
	cudaError_t cycle_grids_transpose(T *first, T *second, T *third, size_t row_count, size_t row_size) {
		cudaError_t cudaStatus;

		dim3 grid(row_count / TILE_SIZE, row_size / TILE_SIZE), threads(TILE_SIZE, BLOCK_ROWS);
		transpose_kernell<TILE_SIZE,BLOCK_ROWS><<<grid, threads>>>(third, second, row_count, row_size);
		if (cudaSuccess != (cudaStatus = cudaGetLastError()))
			return cudaStatus;

		transpose_kernell<TILE_SIZE, BLOCK_ROWS><<<grid, threads>>>(second, first, row_count, row_size);
		if (cudaSuccess != (cudaStatus = cudaGetLastError()))
			return cudaStatus;

		return cudaStatus;
	}
}/* device */ } /* math */ } /* iki */