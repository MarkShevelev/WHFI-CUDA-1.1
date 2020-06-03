#pragma once

#pragma once

#include "DeviceMemory.h"
#include "DeviceDataLine.cuh"

#include <cuda_runtime.h>
#include <algorithm>

namespace iki {	namespace table {
	template <typename T>
	struct HostManagedDeviceDataLine {
		HostManagedDeviceDataLine(unsigned size) : size(size), dMemory(size * sizeof(T)) { }

		HostManagedDeviceDataLine(HostManagedDeviceDataLine const &src) = delete;
		HostManagedDeviceDataLine &operator=(HostManagedDeviceDataLine const &src) = delete;
		HostManagedDeviceDataLine(HostManagedDeviceDataLine &&src) = default;
		HostManagedDeviceDataLine &operator=(HostManagedDeviceDataLine &&src) = default;

		T *data() {
			return (T *)dMemory.get_pointer();
		}

		T const *data() const {
			return (T const *)dMemory.get_pointer();
		}

		device::DeviceDataLine<T> line() const {
			device::DeviceDataLine<T> device_line;
			device_line.size = size;
			device_line.device_ptr = (T *)dMemory.get_pointer();
			return device_line;
		}

		unsigned size;
		DeviceMemory dMemory;
	};
}/*iki*/ }/*table*/
