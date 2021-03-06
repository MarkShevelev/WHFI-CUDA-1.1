#pragma once

#include "DeviceTable.cuh"

#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device {
	template <typename T>
	__device__ T along_mixed_term_discretization(table::device::DeviceTable<T> x_curr, table::device::DeviceTable<T> along_mixed_dfc, unsigned row_idx, unsigned elm_idx) {
		T right_grad = T(0.5) * (x_curr(row_idx + 1, elm_idx + 1) - x_curr(row_idx - 1, elm_idx + 1));
		T central_grad = T(0.5) * (x_curr(row_idx + 1, elm_idx) - x_curr(row_idx - 1, elm_idx));
		T left_grad = T(0.5) * (x_curr(row_idx + 1, elm_idx - 1) - x_curr(row_idx - 1, elm_idx - 1));

		T right_q = along_mixed_dfc(row_idx, elm_idx) * T(0.5) * (right_grad + central_grad);
		T left_q = along_mixed_dfc(row_idx, elm_idx-1) * T(0.5) * (left_grad + central_grad);

		return (right_q - left_q);
	}

	template <typename T>
	__device__ T perp_mixed_term_discretization(table::device::DeviceTable<T> x_curr, table::device::DeviceTable<T> perp_mixed_dfc, unsigned row_idx, unsigned elm_idx) {
		T right_grad = T(0.5) * (x_curr(row_idx + 1, elm_idx + 1) - x_curr(row_idx + 1, elm_idx - 1));
		T central_grad = T(0.5) * (x_curr(row_idx, elm_idx + 1) - x_curr(row_idx, elm_idx - 1));
		T left_grad = T(0.5) * (x_curr(row_idx - 1, elm_idx + 1) - x_curr(row_idx - 1, elm_idx - 1));

		T right_q = perp_mixed_dfc(elm_idx, row_idx) * T(0.5) * (right_grad + central_grad);
		T left_q = perp_mixed_dfc(elm_idx, row_idx - 1) * T(0.5) * (left_grad + central_grad);

		return (right_q - left_q);
	}
}/*device*/ }/*diffusion*/ }/*iki*/