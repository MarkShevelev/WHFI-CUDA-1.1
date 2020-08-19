#pragma once

#include "DeviceTable.cuh"
#include "DeviceDataLine.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace whfi { namespace device { 
	template <typename T>
	__global__ void perp_mean_energy_kernel(table::device::DeviceTable<T> vdf_table, T begin, T step, table::device::DeviceDataLine<T> mean_energy) {
		unsigned row_idx = threadIdx.x + blockDim.x * blockIdx.x;

		T sum = T(0.); //3/8 formula
		for (unsigned elm_idx = 1; elm_idx + 3 < vdf_table.row_size; elm_idx += 3) {
			T 
				arg = sqrt((begin + elm_idx * step)/2)
				, arg_p1 = sqrt((begin + (elm_idx + 1) * step) / 2)
				, arg_p2 = sqrt((begin + (elm_idx + 2) * step) / 2)
				, arg_p3 = sqrt((begin + (elm_idx + 3) * step) / 2)
			;
			sum += T(3. / 8.) * step * (arg * vdf_table(row_idx, elm_idx) + T(3.) * arg_p1 * vdf_table(row_idx, elm_idx + 1) + T(3.) * arg_p2 * vdf_table(row_idx, elm_idx + 2) + arg_p3 * vdf_table(row_idx, elm_idx + 3));
		}
		mean_energy(row_idx) = sum;
	}
}/*device*/ }/*whfi*/ }/*iki*/