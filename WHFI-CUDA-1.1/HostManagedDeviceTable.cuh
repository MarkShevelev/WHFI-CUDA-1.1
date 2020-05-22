#pragma once

#include "DeviceMemory.h"
#include "DeviceTable.cuh"

#include <cuda_runtime.h>

namespace iki { namespace table {
	template <typename T>
	struct HostManagedDeviceTable {
		HostManagedDeviceTable(unsigned row_count, unsigned row_size): dMemory(row_count * row_size * sizeof(T)) {
			dTable.row_count = row_count; 
			dTable.row_size = row_size; 
			dTable.dData = (T*)dMemory.get_pointer();
		}

		HostManagedDeviceTable(HostManagedDeviceTable &&src) = default;
		HostManagedDeviceTable &operator=(HostManagedDeviceTable &&src) = default;
		
		HostManagedDeviceTable& swap_sizes() {
			std::swap(dTable.row_count, dTable.row_size);
			return *this;
		}

		DeviceTable<T> dTable;
		DeviceMemory dMemory;
	};
}/*iki*/ }/*table*/
