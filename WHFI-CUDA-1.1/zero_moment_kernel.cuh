#pragma once

#include "DeviceTable.cuh"
#include "DeviceDataLine.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace whfi { namespace device {
	template <typename T>
	__global__ void zero_moment_kernel(table::device::DeviceTable<T> const vdf_table, T const begin, T const step, table::device::DeviceDataLine<T> zero_moment) {
		unsigned row_idx = threadIdx.x + blockDim.x * blockIdx.x;
 
		T sum = T(0.);
		for (unsigned elm_idx = 2; elm_idx != vdf_table.row_size; elm_idx += 2) {
			sum += step / T(3.) * (vdf_table(row_idx, elm_idx - 1) + T(4.) * vdf_table(row_idx, elm_idx) + vdf_table(row_idx, elm_idx + 1));
		}
		zero_moment(row_idx) = sum;
	}

}/*device*/ }/*whfi*/ }/*iki*/