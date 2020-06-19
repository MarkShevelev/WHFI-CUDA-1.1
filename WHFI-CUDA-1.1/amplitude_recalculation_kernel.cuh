#pragma once

#include "DeviceDataLine.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace whfi { namespace device { 
	template <typename T>
	__global__ void amplitude_recalculation_kernel(table::device::DeviceDataLine<T> const growth_rate, table::device::DeviceDataLine<T> amplitude_spectrum, T dt, T noise_amplitude) {
		unsigned vparall_idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		amplitude_spectrum(vparall_idx) *= 1.0 + 2 * dt * growth_rate(vparall_idx);
		if (amplitude_spectrum(vparall_idx) < noise_amplitude) amplitude_spectrum(vparall_idx) = noise_amplitude;
	}
}/*device*/ }/*whfi*/ }/*iki*/