#pragma once

#include "DeviceTable.cuh"
#include "DeviceDataLine.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace whfi { namespace device {
	template <typename T>
	__global__ void first_moment_kernel(table::device::DeviceTable<T> const vdf_table, T const begin, T const step, table::device::DeviceDataLine<T> zero_moment) {
		unsigned row_idx = threadIdx.x + blockDim.x * blockIdx.x;
 
		T sum = T(0.); //3/8 formula
		for (unsigned elm_idx = 0; elm_idx + 3 < vdf_table.row_size; elm_idx += 3) {
			auto arg1 = begin + step * elm_idx, arg2 = arg1 + step, arg3 = arg2 + step, arg4 = arg3 + step;
			sum += T(3. / 8.) * step * (arg1 * vdf_table(row_idx, elm_idx) + T(3.) * arg2 * vdf_table(row_idx, elm_idx + 1) + T(3.) * arg3 * vdf_table(row_idx, elm_idx + 2) + arg4 * vdf_table(row_idx, elm_idx + 3));
		}
		zero_moment(row_idx) = sum;
	}

}/*device*/ }/*whfi*/ }/*iki*/