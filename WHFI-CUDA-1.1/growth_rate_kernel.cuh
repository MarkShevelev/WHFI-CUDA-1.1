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
		T first_moment_derive = T(0);
		if (0 == idx) {
			first_moment_derive = T(0.5) / vparall_step * (-first_moment(idx + 2) + 4 * first_moment(idx + 1) - 3 * first_moment(idx));
		} else if (first_moment.size - 1 == idx) {
			first_moment_derive = T(0.5) / vparall_step * (first_moment(idx -2) - 4 * first_moment(idx-1) + 3 * first_moment(idx));
		}
		else {
			first_moment_derive = T(0.5) * (first_moment(idx + 1) - first_moment(idx - 1)) / vparall_step;
		}

		//\sqrt(\frac{\pi}{2}) = 1.25331414
		gamma(idx) = -T(1.25331414) * (first_moment_derive - zero_moment(idx) / k_betta(idx)) / dispersion_derive(idx);
	}
}/*device*/ }/*whfi*/ }/*iki*/