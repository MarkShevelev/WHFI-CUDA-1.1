#pragma once

#include "DeviceTable.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace diffusion { namespace device { 
	template <typename T>
	__global__ void along_axis_max_boundary_kernel(table::device::DeviceTable<T> vdf) {
		unsigned row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (0 == row_idx || vdf.row_count - 1 == row_idx) return;
		vdf(row_idx, vdf.row_size - 1) = vdf(row_idx, vdf.row_size - 2);
	}

	template <typename T>
	__global__ void perp_axis_max_boundary_kernel(table::device::DeviceTable<T> vdf) {
		unsigned elm_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (0 == elm_idx || vdf.row_size - 1 == elm_idx) return;
		vdf(vdf.row_count-1,elm_idx) = vdf(vdf.row_count-2, elm_idx);
	}

	template <typename T>
	__global__ void along_axis_min_boundary_kernel(table::device::DeviceTable<T> vdf) {
		unsigned row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (0 == row_idx || vdf.row_count - 1 == row_idx) return;
		vdf(row_idx, 0) = vdf(row_idx, 1);
	}

	template <typename T>
	__global__ void perp_axis_min_boundary_kernel(table::device::DeviceTable<T> vdf) {
		unsigned elm_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (0 == elm_idx || vdf.row_size - 1 == elm_idx) return;
		vdf(0, elm_idx) = vdf(1, elm_idx);
	}
}/*device*/ }/*diffusion*/ }/*iki*/