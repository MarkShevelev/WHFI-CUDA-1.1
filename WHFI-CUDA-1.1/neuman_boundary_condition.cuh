#pragma once

#include "DeviceTable.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace diffusion { namespace device { 
	template <typename T>
	__global__ void along_axis_max_boundary_kernel(table::device::DeviceTable<T> vdf) {
		unsigned row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		vdf(row_idx, vdf.row_size - 1) = vdf(row_idx, vdf.row_size - 2);
	}

	template <typename T>
	__global__ void perp_axis_max_boundary_kernel(table::device::DeviceTable<T> vdf) {
		unsigned elm_idx = threadIdx.x + blockDim.x * blockIdx.x;
		vdf(vdf.row_count-1,elm_idx) = vdf(vdf.row_count-2, elm_idx);
	}

}/*device*/ }/*diffusion*/ }/*iki*/