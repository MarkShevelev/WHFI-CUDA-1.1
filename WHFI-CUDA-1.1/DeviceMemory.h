#pragma once

#include "DeviceError.h"

#include <cuda_runtime.h>

namespace iki {
	class DeviceMemory final {
	public:
		DeviceMemory(unsigned byte_size);

		DeviceMemory(DeviceMemory const &src) = delete;
		DeviceMemory(DeviceMemory &src) = delete;
		DeviceMemory& operator=(DeviceMemory const &src) = delete;
		DeviceMemory& operator=(DeviceMemory &&src) = delete;

		void *get_pointer() const;
		unsigned get_size() const;

		template <typename T>
		explicit operator T* () const { return reinterpret_cast<T *>(device_ptr); }

	private:
		void *device_ptr;
		unsigned byte_size;
	};
} /*iki*/