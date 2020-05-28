#pragma once

#include "DeviceTable.cuh"
#include "diagonal_discretization.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace diffusion { namespace device {
	template <unsigned TILE_SIZE, typename T>
	__global__ void  correction_forward_step_matrix_calculation_kernel(
		table::DeviceTable<T> a, table::DeviceTable<T> b, table::DeviceTable<T> c, table::DeviceTable<T> d,
		table::DeviceTable<T> const x_prev, table::DeviceTable<T> const x_next,
		table::DeviceTable<T> const along_dfc, T along_r) {
		unsigned row_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
		unsigned elm_idx = blockIdx.y * TILE_SIZE + threadIdx.y;

		if (0 == row_idx || x_prev.row_count - 1 == row_idx || 0 == elm_idx || x_prev.row_size - 1 == elm_idx) return;

		a(row_idx,elm_idx) = -0.5 * along_r * along_dfc(row_idx, elm_idx - 1);
		b(row_idx, elm_idx) = T(1) + 0.5 * along_r * (along_dfc(row_idx, elm_idx-1) + along_dfc(row_idx, elm_idx));
		c(row_idx, elm_idx) = -0.5 * along_r * along_dfc(row_idx, elm_idx);
		d(row_idx, elm_idx) = x_next(row_idx, elm_idx)
			- 0.5 * along_r * diagonal_discretization(x_prev(row_idx, elm_idx-1), x_prev(row_idx, elm_idx), x_prev(row_idx, elm_idx+1), along_dfc(row_idx, elm_idx-1), along_dfc(row_idx, elm_idx));
	}
}/*device*/ }/*diffusion*/ }/*iki*/