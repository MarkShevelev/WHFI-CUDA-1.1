#pragma once

#include "DeviceTable.cuh"

#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device {
	template <typename T>
	inline
		__device__ T along_diagonal_discretization(table::device::DeviceTable<T> x_curr, table::device::DeviceTable<T> along_dfc, unsigned row_idx, unsigned elm_idx) {
		return along_dfc(row_idx, elm_idx) * (x_curr(row_idx, elm_idx + 1) - x_curr(row_idx, elm_idx)) - along_dfc(row_idx, elm_idx - 1) * (x_curr(row_idx, elm_idx) - x_curr(row_idx, elm_idx - 1));
	}

	template <typename T>
	inline
		__device__ T perp_diagonal_discretization(table::device::DeviceTable<T> x_curr, table::device::DeviceTable<T> perp_dfc, unsigned row_idx, unsigned elm_idx) {
		return perp_dfc(elm_idx,row_idx) * (x_curr(row_idx+1, elm_idx) - x_curr(row_idx, elm_idx)) - perp_dfc(elm_idx, row_idx - 1) * (x_curr(row_idx, elm_idx) - x_curr(row_idx-1, elm_idx));
	}
}/*device*/ }/*diffusion*/ }/*iki*/