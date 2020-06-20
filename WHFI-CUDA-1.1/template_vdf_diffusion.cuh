#pragma once

#include "PhysicalParameters.h"

#include "HostTable.h"
#include "DeviceTable.cuh"
#include "HostManagedDeviceTable.cuh"
#include "HostDataLine.h"
#include "DeviceDataLine.cuh"
#include "HostManagedDeviceDataLine.cuh"
#include "HostGrid.h"
#include "HostDeviceTransfer.cuh"

#include "VelocityDistributionFunction.h"
#include "initial_diffusion_coefficients_calculation.h"
#include "diffusion_coefficients_recalculation_kernel.cuh"
#include "amplitude_recalculation_kernel.cuh"
#include "neuman_boundary_condition.cuh"

#include "GrowthRateCalculation.cuh"
#include "TwoDimensionalMultithreadDiffusion.cuh"

#include "host_math_helper.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>

std::ostream &exceptional_scientific(std::ostream &ascii_os) {
	ascii_os.exceptions(std::ios::failbit | std::ios::badbit);
	ascii_os.precision(7);
	ascii_os.setf(std::ios::scientific, std::ios::floatfield);
	return ascii_os;
}

namespace iki { namespace whfi {
	template <typename T>
	void vdf_diffusion(
		PhysicalParameters<T> params,
		grid::Space<T> vspace, unsigned vparall_size, unsigned vperp_size,
		T noise_amplitude, T amplitude_amplification_time,
		unsigned iterations, T dt,
		bool log_initial_growth_rate, bool log_result_growth_rate,
		bool log_initial_amplitude, bool log_result_amplitude,
		bool log_initial_diffusion_coefficients
		) {

		grid::Space<T> vspace_transposed = vspace; vspace_transposed.swap_axes();
		auto h_initial_vdf_grid = calculate_muVDF(params, vspace, vparall_size, vperp_size);

		grid::HostGrid<T>
			h_result_vdf_grid(vspace, vparall_size, vperp_size),
			h_core_dfc_vperp_vperp(vspace,vparall_size, vperp_size),
			h_core_dfc_vparall_vparall(vspace_transposed, vperp_size, vparall_size), //transposed
			h_core_dfc_vparall_vperp(vspace,vparall_size, vperp_size),
			h_core_dfc_vperp_vparall(vspace_transposed, vperp_size,vparall_size)     //transposed
		;
		table::HostDataLine<T> h_k_betta(vparall_size), h_dispersion_derive(vparall_size);
		initial_diffusion_coefficients_calculation(
			params, 
			vspace.perp, vspace.along, 
			vparall_size, vperp_size,
			h_core_dfc_vperp_vperp.table,
			h_core_dfc_vparall_vparall.table,     //transposed
			h_core_dfc_vparall_vperp.table,
			h_core_dfc_vperp_vparall.table,       //transposed
			h_k_betta, h_dispersion_derive
		);

		GrowthRateCalculation<T> growth_rate(vspace,h_k_betta,h_dispersion_derive);
		T   vperp_r = dt / math::pow<2u>(vspace.along.step), 
			vparall_r = dt / math::pow<2u>(vspace.perp.step), 
			mixed_r = dt / (vspace.along.step * vspace.perp.step);
		diffusion::TwoDimensionalMultithreadDiffusion<32u, 512u, T> diffusion_solver(
			h_initial_vdf_grid.table,
			h_core_dfc_vperp_vperp.table, vperp_r,
			h_core_dfc_vparall_vparall.table, vparall_r, //transposed
			h_core_dfc_vparall_vperp.table, 
			h_core_dfc_vperp_vparall.table,              //transposed
			mixed_r
		);

		table::HostManagedDeviceTable<T>
			core_dfc_vperp_vperp(table::construct_from(h_core_dfc_vperp_vperp.table)),
			core_dfc_vparall_vparall(table::construct_from(h_core_dfc_vparall_vparall.table)), //transposed
			core_dfc_vparall_vperp(table::construct_from(h_core_dfc_vparall_vperp.table)),
			core_dfc_vperp_vparall(table::construct_from(h_core_dfc_vperp_vparall.table))      //transposed
		;

		growth_rate.recalculate(diffusion_solver.x_prev);

		table::HostDataLine<T> h_growth_rate(table::construct_from(growth_rate.growth_rate));
		//log initial growth rate
		if (log_initial_growth_rate) {
			std::ofstream ascii_os;
			ascii_os << exceptional_scientific;
			ascii_os.open("./data/growth-rate-initial.txt");
			for (unsigned idx = 0; idx != h_growth_rate.size; ++idx) {
				ascii_os << h_k_betta(idx) / params.betta_root_c << ' ' << vspace.perp(idx) << ' ' << h_growth_rate(idx) << '\n';
			}
		}

		table::HostDataLine<T> h_amplitude_spectrum(vparall_size);
		//amplitude premultiplication
		{
			for (unsigned idx = 0; idx != h_amplitude_spectrum.size; ++idx) {
				h_amplitude_spectrum(idx) = h_growth_rate(idx) > T(0.) ? noise_amplitude * std::exp(2 * h_growth_rate(idx) * amplitude_amplification_time) : noise_amplitude;
			}

			if (log_initial_amplitude) {
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/amplitude-initial.txt");
				for (unsigned idx = 0; idx != h_growth_rate.size; ++idx) {
					ascii_os << h_k_betta(idx) / params.betta_root_c << ' ' << vspace.perp(idx) << ' ' << h_amplitude_spectrum(idx) << '\n';
				}
			}
		}
		table::HostManagedDeviceDataLine<T> amplitude_spectrum(table::construct_from(h_amplitude_spectrum));

		//diffusion coefficients multiplication
		{
			cudaError_t cudaStatus;
			whfi::device::diffusion_coefficients_recalculation_kernel<<<vparall_size / 512, 512>>>(
				core_dfc_vperp_vperp.table(),
				core_dfc_vparall_vparall.table(),
				core_dfc_vparall_vperp.table(),
				core_dfc_vperp_vparall.table(),
				amplitude_spectrum.line(),
				diffusion_solver.along_dfc.table(),
				diffusion_solver.perp_dfc.table(),
				diffusion_solver.along_mixed_dfc.table(),
				diffusion_solver.perp_mixed_dfc.table()
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("Diffusion coefficients recalculation kernel failed: ", cudaStatus);
		}

		//log initial diffusion coefficients
		if(false && log_initial_diffusion_coefficients) {
			table::device_to_host_transfer(diffusion_solver.along_dfc, h_core_dfc_vperp_vperp.table);
			table::device_to_host_transfer(diffusion_solver.perp_dfc, h_core_dfc_vparall_vparall.table);
			table::device_to_host_transfer(diffusion_solver.along_mixed_dfc, h_core_dfc_vparall_vperp.table);
			table::device_to_host_transfer(diffusion_solver.perp_mixed_dfc, h_core_dfc_vperp_vparall.table);
			//vperp_vperp
			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/dfc-vperp-vperp.txt");
				ascii_os << h_core_dfc_vperp_vperp;
			}

			//vparall vparall
			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/dfc-vparall-vparall.txt");
				ascii_os << h_core_dfc_vparall_vparall;
			}

			//vparall vperp
			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/dfc-vparall-vperp.txt");
				ascii_os << h_core_dfc_vparall_vperp;
			}

			//vperp vparall
			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/dfc-vperp-vparall.txt");
				ascii_os << h_core_dfc_vperp_vparall;
			}
		}
	
		for (unsigned cnt = 0; cnt != iterations; ++cnt) {
			diffusion_solver.step();
			{
				cudaError_t cudaStatus;
				diffusion::device::perp_axis_max_boundary_kernel<<<vperp_size / 512, 512>>> (diffusion_solver.x_prev.table());
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Boundary condition kernel failed: ", cudaStatus);
			}

			growth_rate.recalculate(diffusion_solver.x_prev);

			{
				cudaError_t cudaStatus;
				whfi::device::amplitude_recalculation_kernel<<<vparall_size / 512, 512>>> (growth_rate.growth_rate.line(), amplitude_spectrum.line(), dt, T(0.));
				cudaDeviceSynchronize();
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Amplitude spectrum recalculation kernel failed: ", cudaStatus);
			}

			//diffusion coefficients multiplication
			if(true) {
				cudaError_t cudaStatus;
				whfi::device::diffusion_coefficients_recalculation_kernel<<<vparall_size / 512, 512>>>(
					core_dfc_vperp_vperp.table(),
					core_dfc_vparall_vparall.table(),
					core_dfc_vparall_vperp.table(),
					core_dfc_vperp_vparall.table(),
					amplitude_spectrum.line(),
					diffusion_solver.along_dfc.table(),
					diffusion_solver.perp_dfc.table(),
					diffusion_solver.along_mixed_dfc.table(),
					diffusion_solver.perp_mixed_dfc.table()
					);
				cudaDeviceSynchronize();
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Diffusion coefficients recalculation kernel failed: ", cudaStatus);
			}
			std::cout << '\r' << cnt;
			if (true && 0 != cnt && 0 == cnt % 1000) {
				std::ostringstream s_os; s_os << "./data/growth-rate-" << cnt << "-intermidiate.txt";
				table::device_to_host_transfer(growth_rate.growth_rate, h_growth_rate);
				{
					std::ofstream ascii_os;
					ascii_os << exceptional_scientific;
					ascii_os.open(s_os.str());
					for (unsigned idx = 0; idx != h_growth_rate.size; ++idx) {
						ascii_os << h_k_betta(idx) / params.betta_root_c << ' ' << vspace.perp(idx) << ' ' << h_growth_rate(idx) << '\n';
					}
				}
			}
		}

		//log result velocity distribution function
		{
			table::device_to_host_transfer(diffusion_solver.x_prev, h_result_vdf_grid.table);
			grid::HostGrid<T> h_result_vdf_ratio_grid(vspace, vparall_size, vperp_size);
			{
				for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx) {
					for (unsigned vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx)
						h_result_vdf_ratio_grid.table(vparall_idx, vperp_idx) = T(1.0) - h_initial_vdf_grid.table(vparall_idx, vperp_idx)/ h_result_vdf_grid.table(vparall_idx, vperp_idx);
				}
			}
			//vdf 
			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/vdf-result.txt");
				ascii_os << h_result_vdf_grid;
			}

			//vdf ratio
			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/vdf-ratio-result.txt");
				ascii_os << h_result_vdf_ratio_grid;
			}
		}

		//log result growth rate
		if(log_result_growth_rate) {
			table::device_to_host_transfer(growth_rate.growth_rate, h_growth_rate);
			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/growth-rate-result.txt");
				for (unsigned idx = 0; idx != h_growth_rate.size; ++idx) {
					ascii_os << h_k_betta(idx) / params.betta_root_c << ' ' << vspace.perp(idx) << ' ' << h_growth_rate(idx) << '\n';
				}
			}
		}

		//log result amplitude
		if (log_result_amplitude) {
			table::device_to_host_transfer(amplitude_spectrum, h_amplitude_spectrum);
			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/amplitude-spectrum-result.txt");
				for (unsigned idx = 0; idx != h_growth_rate.size; ++idx) {
					ascii_os << h_k_betta(idx) / params.betta_root_c << ' ' << vspace.perp(idx) << ' ' << h_amplitude_spectrum(idx) << '\n';
				}
			}
		}

		//log result diffusion coefficients
	}

}/*whfi*/ }/*iki*/
