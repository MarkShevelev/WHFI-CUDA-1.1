#pragma once

#include "DeviceError.h"
#include "HostManagedDeviceTable.cuh"
#include "transpose.cuh"

#include <cuda_runtime.h>

namespace iki { namespace table {
	template <typename T>
	void transpose(HostManagedDeviceTable<T> const &from, HostManagedDeviceTable<T> &to) {
		cudaError_t cudaStatus;
		dim3 grid(from.row_count / 32, from.row_size / 32), threads(32, 8);
		math::device::transpose_kernell<32,8><<<grid,threads>>> (to.data(), from.data(), from.row_count, from.row_size);
		if (cudaSuccess != (cudaStatus = cudaGetLastError()))
			throw DeviceError("Can't transpose: ", cudaStatus);
	}
}/*table*/ }/*iki*/