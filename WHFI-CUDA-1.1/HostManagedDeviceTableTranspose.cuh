#pragma once

#include "DeviceError.h"
#include "HostManagedDeviceTable.cuh"
#include "transpose.cuh"

#include <cuda_runtime.h>

namespace iki { namespace table { namespace test {
	template <typename T>
	void transpose(HostManagedDeviceTable<T> const &from, HostManagedDeviceTable<T> &to) {
		cudaError_t cudaStatus;
		dim3 grid(from.dTable.row_count / 32, from.dTable.row_size / 32), threads(32, 8);
		math::device::transpose_kernell<32,8><<<grid,threads>>> (to.data(), from.data(), from.dTable.row_count, from.dTable.row_size);
		if (cudaSuccess != (cudaStatus = cudaGetLastError()))
			throw DeviceError("Can't transpose: ", cudaStatus);
	}
}/*test*/ }/*table*/ }/*iki*/