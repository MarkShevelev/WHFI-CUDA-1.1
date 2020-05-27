#pragma once

#include "HostTable.h"
#include "HostManagedDeviceTable.cuh"
#include "DeviceError.h"

#include <cuda_runtime.h>

namespace iki { namespace table { namespace test { 
	template <typename T>
	void host_to_device_transfer(HostTable<T> const &host, HostManagedDeviceTable<T> &device) {
		cudaError_t cudaStatus = cudaMemcpy(device.dMemory, host.hData.data(), host.full_size() * sizeof(T), cudaMemcpyHostToDevice);
		if (cudaSuccess != cudaStatus)
			throw DeviceError("Host to device data transfer failed: ", cudaStatus);
	}

	template <typename T>
	void device_to_host_transfer(HostManagedDeviceTable<T> const &device, HostTable<T> &host) {
		cudaError_t cudaStatus = cudaMemcpy(host.hData.data(), device.dMemory, device.full_size() * sizeof(T), cudaMemcpyDeviceToHost);
		if (cudaSuccess != cudaStatus)
			throw DeviceError("Host to device data transfer failed: ", cudaStatus);
	}

	template <typename T>
	HostManagedDeviceTable<T> construct_from(HostTable<T> const &host) {
		HostManagedDeviceTable<T> device_table(host.row_count, host.row_size);
		host_to_device_transfer(host, device_table);
		return device_table;
	}

	template <typename T>
	HostTable<T> construct_from(HostManagedDeviceTable<T> const &device) {
		HostTable<T> host_table(device.dTable.row_count, device.dTable.row_size);
		device_to_host_transfer(device, host_table);
		return host_table;
	}
}/*test*/ }/*table*/ }/*iki*/