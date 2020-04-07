#include "DeviceError.h"

namespace iki {
	DeviceError::DeviceError(std::string const &additional_text, cudaError_t cudaStatus): runtime_error(additional_text + cudaGetErrorString(cudaStatus)) {  }

	DeviceError::DeviceError(cudaError_t cudaStatus): runtime_error(cudaGetErrorString(cudaStatus)) {  }
} /*iki*/