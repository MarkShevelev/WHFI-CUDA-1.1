#pragma once

#include "DeviceTable.cuh"
#include "mixed_term_discretization.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace diffusion { namespace device {
	template <unsigned TILE_SIZE, typename T>
	__global__ void  compensation_matrix_calculation_kernel(
		table::device::DeviceTable<T> d,
		table::device::DeviceTable<T> const x_prev, table::device::DeviceTable<T> const x_next,
		table::device::DeviceTable<T> const along_mixed_dfc, table::device::DeviceTable<T> const perp_mixed_dfc, T mixed_r
	) {
		unsigned row_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
		unsigned elm_idx = blockIdx.y * TILE_SIZE + threadIdx.y;

		if (0 == row_idx || x_prev.row_count - 1 == row_idx || 0 == elm_idx || x_prev.row_size - 1 == elm_idx) return;

		d(row_idx, elm_idx) +=
			0.5 * mixed_r * along_mixed_term_discretization(x_next, along_mixed_dfc, row_idx, elm_idx)
			+ 0.5 * mixed_r * perp_mixed_term_discretization(x_next, perp_mixed_dfc, row_idx, elm_idx)
			- 0.5 * mixed_r * along_mixed_term_discretization(x_prev, along_mixed_dfc, row_idx, elm_idx)
			- 0.5 * mixed_r * perp_mixed_term_discretization(x_prev, perp_mixed_dfc, row_idx, elm_idx);
	}
}/*device*/ }/*diffusion*/ }/*iki*/