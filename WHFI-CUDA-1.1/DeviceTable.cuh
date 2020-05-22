#pragma once

#include <cuda_runtime.h>

namespace iki { namespace table {
	template <typename T>
	struct DeviceTable {
		__device__ T operator()(unsigned row_idx, unsigned elm_idx) const {
			return dData[row_idx + elm_idx * row_count];
		}

		__device__ T& operator()(unsigned row_idx, unsigned elm_idx) {
			return dData[row_idx + elm_idx * row_count];
		}

		unsigned row_count, row_size;
		T *dData;
	};
}/*table*/ }/*iki*/