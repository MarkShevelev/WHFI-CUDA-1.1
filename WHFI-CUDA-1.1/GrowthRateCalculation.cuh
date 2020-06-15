#pragma once

#include "DeviceError.h"
#include "DeviceTable.cuh"
#include "HostManagedDeviceTable.cuh"
#include "DeviceDataLine.cuh"
#include "HostManagedDeviceDataLine.cuh"
#include "HostTable.h"
#include "HostDataLine.h"
#include "HostGrid.h"
#include "HostDeviceTransfer.cuh"
#include "zero_moment_kernel.cuh"
#include "first_moment_kernel.cuh"
#include "growth_rate_kernel.cuh"


namespace iki { namespace whfi { 
	template <typename T>
	struct GrowthRateCalculation {
		GrowthRateCalculation(
			grid::Space<T> vspace,
			table::HostDataLine<T> const &h_k_betta,
			table::HostDataLine<T> const &h_dispersion_derive
		): 
		vspace(vspace), 
		k_betta(table::construct_from(h_k_betta)),
		dispersion_derive(table::construct_from(h_dispersion_derive)),
		zero_moment(h_k_betta.size),
		first_moment(h_k_betta.size),
		growth_rate(h_k_betta.size)
		{ }

		void recalculate(table::HostManagedDeviceTable<T> const &vdf_table) {
			cudaError_t cudaStatus;

			dim3 threads(512), blocks((vdf_table.row_count + threads.x - 1) / threads.x);
			device::zero_moment_kernel<<<blocks, threads>>>(
				vdf_table.table(),
				vspace.along.begin, vspace.along.step, 
				zero_moment.line()
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("GrowthRate recalculation.zero_moment: ", cudaStatus);

			device::first_moment_kernel<<<blocks, threads>>>(
				vdf_table.table(),
				vspace.along.begin, vspace.along.step,
				first_moment.line()
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("GrowthRate recalculation.first_moment: ", cudaStatus);

			device::growth_rate_kernel<<<blocks, threads>>>(
				zero_moment.line(),
				first_moment.line(),
				k_betta.line(),
				dispersion_derive.line(),
				vspace.perp.step,
				growth_rate.line()
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("GrowthRate recalculation.growth_rate: ", cudaStatus);
		}

		grid::Space<T> const vspace;
		table::HostManagedDeviceDataLine<T> const k_betta, dispersion_derive;
		table::HostManagedDeviceDataLine<T> zero_moment, first_moment, growth_rate;
	};
}/*whfi*/ }/*iki*/