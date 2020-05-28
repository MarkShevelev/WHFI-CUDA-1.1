#pragma once

#include <cuda_runtime.h>

namespace iki { namespace table { namespace device {
	template <typename T>
	struct DeviceTable {
		__device__ T operator()(unsigned row_idx, unsigned elm_idx) const {
			return device_ptr[row_idx + elm_idx * row_count];
		}

		__device__ T& operator()(unsigned row_idx, unsigned elm_idx) {
			return device_ptr[row_idx + elm_idx * row_count];
		}

		__device__ unsigned full_size() const {
			return row_count * row_size;
		}

		unsigned row_count, row_size;
		T *device_ptr;
	};
}/*device*/ }/*table*/ }/*iki*/