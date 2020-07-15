#pragma once

#include "DeviceDataLine.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace whfi { namespace device { 
	template <typename T>
	__global__ void growth_rate_kernel(
		iki::table::device::DeviceDataLine<T> const zero_moment,
		iki::table::device::DeviceDataLine<T> const first_moment,
		iki::table::device::DeviceDataLine<T> const k_betta,
		iki::table::device::DeviceDataLine<T> const dispersion_derive,
		T const vparall_step,
		iki::table::device::DeviceDataLine<T> gamma
	) {
		unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (0 == idx || first_moment.size - 1 == idx) return;
		
		T first_moment_derive;
		if (1 == idx)
			first_moment_derive = (first_moment(idx + 1) - first_moment(idx)) / vparall_step;
		else
			if (first_moment.size - 2 == idx)
				first_moment_derive = (first_moment(idx) - first_moment(idx - 1)) / vparall_step;
			else
				first_moment_derive = T(0.5) * (first_moment(idx + 1) - first_moment(idx - 1)) / vparall_step;

		//\sqrt(\frac{\pi}{2}) = 1.25331414
		auto gr =  -T(1.25331414) * (first_moment_derive - zero_moment(idx) / k_betta(idx)) / dispersion_derive(idx);

		gamma(idx) = gr;
		if (1 == idx) { gamma(0) = gr; }
		if (first_moment.size - 2 == idx) { gamma(first_moment.size - 1) = gr; }
	}
}/*device*/ }/*whfi*/ }/*iki*/