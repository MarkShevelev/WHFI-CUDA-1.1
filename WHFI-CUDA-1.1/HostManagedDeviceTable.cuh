#pragma once

#include "DeviceMemory.h"
#include "DeviceTable.cuh"

#include <cuda_runtime.h>
#include <algorithm>

namespace iki { namespace table {
	template <typename T>
	struct HostManagedDeviceTable {
		HostManagedDeviceTable(unsigned row_count, unsigned row_size): row_count(row_count), row_size(row_size), dMemory(row_count * row_size * sizeof(T)) {	}

		HostManagedDeviceTable(HostManagedDeviceTable const &src) = delete;
		HostManagedDeviceTable& operator=(HostManagedDeviceTable const &src) = delete;
		HostManagedDeviceTable(HostManagedDeviceTable &&src) = default;
		HostManagedDeviceTable& operator=(HostManagedDeviceTable &&src) = default;
		
		HostManagedDeviceTable& swap_sizes() {
			std::swap(row_count, row_size);
			return *this;
		}

		unsigned full_size() const {
			return row_count * row_size;
		}

		T *data() {
			return (T *)dMemory.get_pointer();
		}

		T const *data() const {
			return (T const *)dMemory.get_pointer();
		}

		device::DeviceTable<T> table() const {
			device::DeviceTable<T> device_table;
			device_table.row_count = row_count;
			device_table.row_size = row_size;
			device_table.device_ptr = (T *)dMemory.get_pointer();
			return device_table;
		}

		unsigned row_count, row_size;
		DeviceMemory dMemory;
	};
}/*iki*/ }/*table*/
