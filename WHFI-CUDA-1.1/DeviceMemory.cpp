#include "DeviceMemory.h"

namespace iki {
	DeviceMemory::DeviceMemory(unsigned byte_size) : device_ptr(nullptr), byte_size(0u) {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaMalloc(&device_ptr, byte_size)))
			throw DeviceError("Can't allocate memory.\nCause: ", cudaStatus);
		this->byte_size = byte_size;
	}

	void* DeviceMemory::get_pointer() const { return device_ptr; }
	unsigned DeviceMemory::get_size() const { return byte_size; }
}/*iki*/