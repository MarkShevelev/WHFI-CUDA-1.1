#pragma once

#include "HostTable.h"
#include "HostManagedDeviceTable.cuh"
#include "HostDataLine.h"
#include "HostManagedDeviceDataLine.cuh"
#include "DeviceError.h"

#include <cuda_runtime.h>

namespace iki { namespace table {
	template <typename T>
	void host_to_device_transfer(HostTable<T> const &host, HostManagedDeviceTable<T> &device) {
		cudaError_t cudaStatus = cudaMemcpy(device.dMemory, host.data(), host.full_size() * sizeof(T), cudaMemcpyHostToDevice);
		if (cudaSuccess != cudaStatus)
			throw DeviceError("Host to device data transfer failed: ", cudaStatus);
	}

	template <typename T>
	void device_to_host_transfer(HostManagedDeviceTable<T> const &device, HostTable<T> &host) {
		cudaError_t cudaStatus = cudaMemcpy(host.data(), device.dMemory, device.full_size() * sizeof(T), cudaMemcpyDeviceToHost);
		if (cudaSuccess != cudaStatus)
			throw DeviceError("Device to host data transfer failed: ", cudaStatus);
	}

	template <typename T>
	HostManagedDeviceTable<T> construct_from(HostTable<T> const &host) {
		HostManagedDeviceTable<T> device_table(host.row_count, host.row_size);
		host_to_device_transfer(host, device_table);
		return device_table;
	}

	template <typename T>
	HostTable<T> construct_from(HostManagedDeviceTable<T> const &device) {
		HostTable<T> host_table(device.row_count, device.row_size);
		device_to_host_transfer(device, host_table);
		return host_table;
	}

	template <typename T>
	void host_to_device_transfer(HostDataLine<T> const &host, HostManagedDeviceDataLine<T> &device) {
		cudaError_t cudaStatus = cudaMemcpy(device.dMemory, host.data(), host.size * sizeof(T), cudaMemcpyHostToDevice);
		if (cudaSuccess != cudaStatus)
			throw DeviceError("Host to device data transfer failed: ", cudaStatus);
	}

	template <typename T>
	void device_to_host_transfer(HostManagedDeviceDataLine<T> const &device, HostDataLine<T> &host) {
		cudaError_t cudaStatus = cudaMemcpy(host.data(), device.dMemory, device.size * sizeof(T), cudaMemcpyDeviceToHost);
		if (cudaSuccess != cudaStatus)
			throw DeviceError("Device to host data transfer failed: ", cudaStatus);
	}

	template <typename T>
	HostManagedDeviceDataLine<T> construct_from(HostDataLine<T> const &host) {
		HostManagedDeviceDataLine<T> device_line(host.size);
		host_to_device_transfer(host, device_line);
		return device_line;
	}

	template <typename T>
	HostDataLine<T> construct_from(HostManagedDeviceDataLine<T> const &device) {
		HostDataLine<T> host_line(device.size);
		device_to_host_transfer(device, host_line);
		return host_line;
	}
}/*table*/ }/*iki*/