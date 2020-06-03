#pragma once

#include <cuda_runtime.h>

namespace iki { namespace table { namespace device {
	template <typename T>
	struct DeviceDataLine {
		__device__ T operator()(unsigned idx) const {
			return device_ptr[idx];
		}

		__device__ T& operator()(unsigned idx)  {
			return device_ptr[idx];
		}

		unsigned size;
		T *device_ptr;
	};
}/*device*/ }/*table*/ }/*iki*/