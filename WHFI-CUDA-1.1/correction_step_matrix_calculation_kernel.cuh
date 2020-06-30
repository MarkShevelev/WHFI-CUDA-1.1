#pragma once

#include "DeviceTable.cuh"
#include "diagonal_discretization.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace diffusion { namespace device {
	template <unsigned TILE_SIZE, typename T>
	__global__ void  correction_step_matrix_calculation_kernel(
		table::device::DeviceTable<T> a, table::device::DeviceTable<T> b, table::device::DeviceTable<T> c, table::device::DeviceTable<T> d,
		table::device::DeviceTable<T> const x_prev, table::device::DeviceTable<T> const x_next,
		table::device::DeviceTable<T> const along_dfc, T along_r, 
		table::device::DeviceTable<T> const perp_mixed_dfc, T mixed_r
	) {
		unsigned row_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
		unsigned elm_idx = blockIdx.y * TILE_SIZE + threadIdx.y;

		if (0 == row_idx || x_prev.row_count - 1 == row_idx || 0 == elm_idx || x_prev.row_size - 1 == elm_idx) return;

		T adv_row = T(0.25) * (perp_mixed_dfc(row_idx + 1, elm_idx) - perp_mixed_dfc(row_idx - 1, elm_idx));
		a(row_idx,elm_idx) = -T(0.5) * (along_r * along_dfc(row_idx, elm_idx - 1) - mixed_r * adv_row);
		b(row_idx, elm_idx) = T(1) + T(0.5) * along_r * (along_dfc(row_idx, elm_idx-1) + along_dfc(row_idx, elm_idx));
		c(row_idx, elm_idx) = -T(0.5) * (along_r * along_dfc(row_idx, elm_idx) + mixed_r * adv_row);
		d(row_idx, elm_idx) = x_next(row_idx, elm_idx)
			- T(0.5) * along_r * along_diagonal_discretization(x_prev, along_dfc, row_idx, elm_idx)
			- T(0.5) * mixed_r * adv_row * (x_prev(row_idx, elm_idx + 1) - x_prev(row_idx, elm_idx - 1));
	}
}/*device*/ }/*diffusion*/ }/*iki*/