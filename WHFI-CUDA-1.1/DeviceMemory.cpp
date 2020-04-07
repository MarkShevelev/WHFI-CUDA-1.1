#include "DeviceMemory.h"

#include <algorithm>

using namespace std;

namespace iki {
	DeviceMemory::DeviceMemory(unsigned byte_size) : device_ptr(nullptr), byte_size(0u) {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaMalloc(&device_ptr, byte_size)))
			throw DeviceError("Can't allocate memory.\nCause: ", cudaStatus);
		this->byte_size = byte_size;
	}

	DeviceMemory::DeviceMemory(DeviceMemory const &src): DeviceMemory(src.byte_size) {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaMemcpy(device_ptr, src.device_ptr, byte_size, cudaMemcpyDeviceToDevice)))
			throw DeviceError("Can't copy memory from one array into another on device.\nCause: ", cudaStatus);
	}

	DeviceMemory& DeviceMemory::operator=(DeviceMemory const &src) {
		DeviceMemory tmp(src);
		swap(tmp.device_ptr, device_ptr);
		swap(tmp.byte_size, byte_size);
		return *this;
	}

	DeviceMemory::DeviceMemory(DeviceMemory &&src) : device_ptr(src.device_ptr), byte_size(src.byte_size) {
		src.device_ptr = nullptr;
		src.byte_size = 0;
	}

	DeviceMemory& DeviceMemory::operator=(DeviceMemory &&src) {
		DeviceMemory tmp(move(src));
		swap(tmp.device_ptr, device_ptr);
		swap(tmp.byte_size, byte_size);
		return *this;
	}

	void* DeviceMemory::get_pointer() const { return device_ptr; }
	unsigned DeviceMemory::get_size() const { return byte_size; }
}/*iki*/