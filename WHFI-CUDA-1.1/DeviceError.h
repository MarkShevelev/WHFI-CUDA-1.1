#pragma once

#include <cuda_runtime.h>

#include <string>
#include <stdexcept>

namespace iki {
	class DeviceError final : public std::runtime_error {
	public:
		DeviceError(std::string const &additional_text, cudaError_t cudaStatus);
		DeviceError(cudaError_t cudaStatus);
	};
} /*iki*/