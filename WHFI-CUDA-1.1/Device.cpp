#include "Device.h"
#include "DeviceError.h"

#include <cuda_runtime.h>

#include <stdexcept>

using namespace std;

namespace iki {
	Device::Device(int device) {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaSetDevice(device)))
			throw DeviceError(cudaStatus);
	}

	Device::~Device() noexcept {
		cudaDeviceReset();
	}
} /*iki*/