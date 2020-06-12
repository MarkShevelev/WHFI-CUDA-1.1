#pragma once

#include "DeviceTable.cuh"
#include "DeviceDataLine.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace whfi { namespace device {
	template <typename T>
	__global__ void diffusion_coefficients_recalculation_kernel(
		table::device::DeviceTable<T> const core_dfc_vperp_vperp,
		table::device::DeviceTable<T> const core_dfc_vparall_vparall,    //transposed
		table::device::DeviceTable<T> const core_dfc_vparall_vperp,
		table::device::DeviceTable<T> const core_dfc_vperp_vparall,      //transposed
		table::device::DeviceDataLine<T> const amplitude_spectrum,
		table::device::DeviceTable<T> dfc_vperp_vperp,
		table::device::DeviceTable<T> dfc_vparall_vparall,    //transposed
		table::device::DeviceTable<T> dfc_vparall_vperp,
		table::device::DeviceTable<T> dfc_vperp_vparall       //transposed
		
	) {
		unsigned row_idx = threadIdx.x + blockIdx.x * blockDim.x;

		auto coeff = amplitude_spectrum(row_idx);
		for (unsigned elm_idx = 0; elm_idx != dfc_vperp_vperp.row_size; ++elm_idx) {
			dfc_vperp_vperp(row_idx, elm_idx) = core_dfc_vperp_vperp(row_idx, elm_idx) * coeff;
			dfc_vparall_vparall(elm_idx, row_idx) = core_dfc_vparall_vparall(elm_idx, row_idx) * coeff;
			dfc_vparall_vperp(row_idx, elm_idx) = core_dfc_vparall_vperp(row_idx, elm_idx) * coeff;
			dfc_vperp_vparall(elm_idx, row_idx) = core_dfc_vperp_vparall(elm_idx, row_idx) * coeff;
		}
	}
}/*device*/ }/*whfi*/ }/*iki*/