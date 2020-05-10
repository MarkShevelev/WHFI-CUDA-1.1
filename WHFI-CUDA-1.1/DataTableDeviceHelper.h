#pragma once

#include "DataTable.h"
#include "DeviceError.h"

#include <cuda_runtime.h>

namespace iki { namespace table { namespace device {
	template <typename T, unsigned Dim, unsigned Scale>
	T* to_device(T *device_ptr, DataTable<T, Dim, Scale> const &table) {
		cudaError_t cudaStatus = cudaMemcpy(device_ptr, table.raw_data(), sizeof(T) * index_volume(table.get_bounds()) * Scale, cudaMemcpyHostToDevice);
		if (cudaSuccess != cudaStatus)
			throw DeviceError(cudaStatus);
		return device_ptr;
	}

	template <typename T, unsigned Dim, unsigned Scale>
	DataTable<T, Dim, Scale>& from_device(DataTable<T, Dim, Scale> &table, T const *device_ptr) {
		cudaError_t cudaStatus = cudaMemcpy(table.raw_data(), device_ptr, sizeof(T) * index_volume(table.get_bounds()) * Scale, cudaMemcpyDeviceToHost);
		if (cudaSuccess != cudaStatus)
			throw DeviceError(cudaStatus);
		return table;
	}
}/*device*/ }/*table*/ }/*iki*/