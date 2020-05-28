#pragma once

#include "DeviceTable.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace math { namespace device { 
	template <typename T>
	__global__ void thomson_sweep_kernel(
		table::DeviceTable<T> a, table::DeviceTable<T> b, table::DeviceTable<T> c, table::DeviceTable<T> d,
		table::DeviceTable<T> x_curr) {
		unsigned row_idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (0 == row_idx || x_curr.row_count - 1 == row_idx) return;

		for (size_t elm_idx = 2; elm_idx != x_curr.row_size-1; ++elm_idx) {
			T w = a(row_idx,elm_idx) / b(row_idx,elm_idx-1);
			b(row_idx, elm_idx) = fma(-w, c(row_idx, elm_idx-1), b(row_idx, elm_idx));
			d(row_idx, elm_idx) = fma(-w, d(row_idx, elm_idx-1), d(row_idx, elm_idx));
		}
		x_curr(row_idx,x_curr.row_size-2) = d(row_idx, x_curr.row_size - 2) / b(row_idx, x_curr.row_size - 2);

		for (size_t elm_idx = x_curr.row_size - 3; elm_idx != 0; --elm_idx) {
			x_curr(row_idx,elm_idx) = fma(-c(row_idx, elm_idx), x_curr(row_idx, elm_idx+1), d(row_idx, elm_idx)) / b(row_idx, elm_idx);
		}
	}
}/*device*/ }/*math*/ }/*iki*/