#pragma once

#include "DeviceError.h"
#include "HostManagedDeviceTable.cuh"
#include "transpose.cuh"

#include <cuda_runtime.h>

namespace iki { namespace table { namespace test {
	template <typename T>
	void transope(HostManagedDeviceTable<T> const &from, HostManagedDeviceTable<T> &to) {
		cudaError_t cudaStatus;

		dim3 grid(from.row_count / 32, from.row_size / 32), threads(32, 8);
		math::device::transpose_kernell<32,8><<<grid,threads>>> (to.data(), from.data(), from.dTable.row_count, from.dTable.row_size);
		if (cudaSuccess != (cudaStatus = cudaGetLastError()))
			throw DeviceError("Can't transpose: ", cudaStatus);
		to.swap_sizes();
	}

	template <typename T>
	void cycle_transpose(HostManagedDeviceTable<T> &t1, HostManagedDeviceTable<T> &t2, HostManagedDeviceTable<T> &t3) { //t1->t2->t3
		transpose(t2, t3);
		transpose(t1, t3);
	}
}/*test*/ }/*table*/ }/*iki*/