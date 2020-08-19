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

#include "data_line_folding.h"
#include "perp_mean_energy.cuh"

#include <vector>
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
		bool log_initial_diffusion_coefficients,
		bool dfc_recalculation,
		bool log_growth_rate_intermidiate, unsigned gr_log_period,
		bool log_vdf_intermidiate, unsigned vdf_log_period,
		unsigned vparall_rate, unsigned vperp_rate
		) {

		grid::Space<T> vspace_transposed = vspace; vspace_transposed.swap_axes();
		auto h_initial_vdf_grid = calculate_muVDF(params, vspace, vparall_size, vperp_size);
		for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx)
			h_initial_vdf_grid.table(vparall_idx, 0) = h_initial_vdf_grid.table(vparall_idx, 1);

		grid::HostGrid<T>
			h_result_vdf_grid(vspace, vparall_size, vperp_size),
			h_result_vdf_diff_grid(vspace, vparall_size, vperp_size),
			h_core_dfc_vperp_vperp(vspace,vparall_size, vperp_size),
			h_core_dfc_vparall_vparall(vspace_transposed, vperp_size, vparall_size), //transposed
			h_core_dfc_vparall_vperp(vspace,vparall_size, vperp_size),
			h_core_dfc_vperp_vparall(vspace_transposed, vperp_size,vparall_size)     //transposed
		;
		table::HostDataLine<T> h_k_betta(vparall_size), h_dispersion_derive(vparall_size), h_growth_rate(vparall_size), h_amplitude(vparall_size);
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

		//log initial diffusion coefficients
		if (log_initial_diffusion_coefficients) {
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
		//log initial growth rate and amplitude
		{
			table::device_to_host_transfer(growth_rate.growth_rate, h_growth_rate);
			//amplitude premultiplication
			for (unsigned idx = 0; idx != h_amplitude.size; ++idx) {
				h_amplitude(idx) = h_growth_rate(idx) > 0 ? noise_amplitude * std::exp(2 * h_growth_rate(idx) * amplitude_amplification_time) : T(0);
			}

			std::ofstream ascii_os;
			ascii_os << exceptional_scientific;
			ascii_os.open("./data/growth-rate-initial.txt");
			for (unsigned idx = 0; idx != h_growth_rate.size; ++idx) {
				ascii_os << h_k_betta(idx) / params.betta_root_c << ' ' << vspace.perp(idx) << ' ' << h_growth_rate(idx) << ' ' << h_amplitude(idx) << '\n';
			}
		}

		table::HostManagedDeviceDataLine<T> amplitude(table::construct_from(h_amplitude));
		//diffusion coefficients multiplication
		{
			cudaError_t cudaStatus;
			whfi::device::diffusion_coefficients_recalculation_kernel<<<vparall_size / 512, 512>>>(
				core_dfc_vperp_vperp.table(),
				core_dfc_vparall_vparall.table(),
				core_dfc_vparall_vperp.table(),
				core_dfc_vperp_vparall.table(),
				amplitude.line(),
				diffusion_solver.along_dfc.table(),
				diffusion_solver.perp_dfc.table(),
				diffusion_solver.along_mixed_dfc.table(),
				diffusion_solver.perp_mixed_dfc.table()
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("Diffusion coefficients recalculation kernel failed: ", cudaStatus);
		}
		//vdf foled history
		table::HostDataLine<T> h_zero_moment(vparall_size);
		std::vector<T> vdf_folded_history(iterations);

		//vdf mean perpendicular energy
		table::HostManagedDeviceDataLine<T> d_mean_perp_energy_vparall(vparall_size);
		table::HostDataLine<T> h_mean_perp_energy_vparall(vparall_size);
		std::vector<T> vdf_mean_perp_energy_history(iterations);


		for (unsigned cnt = 0; cnt != iterations; ++cnt) {
			diffusion_solver.step();
			std::cout << '\r' << cnt;

			//boundary conditions
			{
				cudaError_t cudaStatus;
				diffusion::device::perp_axis_max_boundary_kernel<<<vperp_size / 512, 512>>> (diffusion_solver.x_prev.table());
				diffusion::device::along_axis_min_boundary_kernel<<<vparall_size / 512, 512>>> (diffusion_solver.x_prev.table());
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Boundary condition kernel failed: ", cudaStatus);
			}

			//growth rate and amplitude recalculation
			growth_rate.recalculate(diffusion_solver.x_prev);
			if(dfc_recalculation){
				cudaError_t cudaStatus;
				whfi::device::amplitude_recalculation_kernel<<<vparall_size / 512, 512>>> (growth_rate.growth_rate.line(), amplitude.line(), dt, T(0.));
				cudaDeviceSynchronize();
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Amplitude spectrum recalculation kernel failed: ", cudaStatus);

				whfi::device::diffusion_coefficients_recalculation_kernel<<<vparall_size / 512, 512>>>(
					core_dfc_vperp_vperp.table(),
					core_dfc_vparall_vparall.table(),
					core_dfc_vparall_vperp.table(),
					core_dfc_vperp_vparall.table(),
					amplitude.line(),
					diffusion_solver.along_dfc.table(),
					diffusion_solver.perp_dfc.table(),
					diffusion_solver.along_mixed_dfc.table(),
					diffusion_solver.perp_mixed_dfc.table()
					);
				cudaDeviceSynchronize();
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Diffusion coefficients recalculation kernel failed: ", cudaStatus);
			}

			//vdf folded history
			device_to_host_transfer(growth_rate.zero_moment, h_zero_moment);
			vdf_folded_history[cnt] = data_line_folding(h_zero_moment, vspace.perp.step);

			//vdf mean perp energy history
			{

				device::perp_mean_energy_kernel<<<(vparall_size - 1) / 512 + 1, 512>>> (diffusion_solver.x_prev.table(), vspace.along.begin, vspace.along.step, d_mean_perp_energy_vparall.line());
				device_to_host_transfer(d_mean_perp_energy_vparall, h_mean_perp_energy_vparall);
				vdf_mean_perp_energy_history[cnt] = data_line_folding(h_mean_perp_energy_vparall, vspace.perp.step);
			}

			if (log_growth_rate_intermidiate && 0 != cnt && 0 == cnt % gr_log_period) {
				std::ostringstream s_os; s_os << "./data/growth-rate-" << cnt << "-intermidiate.txt";
				table::device_to_host_transfer(growth_rate.growth_rate, h_growth_rate);
				table::device_to_host_transfer(amplitude, h_amplitude);
				{
					std::ofstream ascii_os;
					ascii_os << exceptional_scientific;
					ascii_os.open(s_os.str());
					for (unsigned idx = 0; idx != h_growth_rate.size; ++idx) {
						ascii_os << h_k_betta(idx) / params.betta_root_c << ' ' << vspace.perp(idx) << ' ' << h_growth_rate(idx) << ' ' << h_amplitude(idx) << '\n';
					}
				}
			}
			
			if (log_vdf_intermidiate && 0 != cnt && 0 == cnt % vdf_log_period) {
				std::ostringstream vdf_sos; vdf_sos << "./data/vdf-" << cnt << "-intermidiate.txt";
				table::device_to_host_transfer(diffusion_solver.x_prev, h_result_vdf_grid.table);
				for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx) {
					for (unsigned vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx)
						h_result_vdf_diff_grid.table(vparall_idx, vperp_idx) = h_result_vdf_grid.table(vparall_idx, vperp_idx) - h_initial_vdf_grid.table(vparall_idx, vperp_idx);
				}

				{
					std::ofstream ascii_os;
					ascii_os << exceptional_scientific;
					ascii_os.open(vdf_sos.str());
					for (unsigned vparall_idx = 0; vparall_idx < vparall_size; vparall_idx += vparall_rate) {
						for (unsigned vperp_idx = 0; vperp_idx < vperp_size; vperp_idx += vperp_rate)
							ascii_os << vspace.perp(vparall_idx) << ' ' << vspace.along(vperp_idx) << ' ' << std::sqrt(2 * std::fabs(vspace.along(vperp_idx))) << ' ' << h_result_vdf_grid.table(vparall_idx, vperp_idx) << ' ' << h_initial_vdf_grid.table(vparall_idx, vperp_idx) << ' ' << h_result_vdf_diff_grid.table(vparall_idx, vperp_idx) << ' ' << h_result_vdf_diff_grid.table(vparall_idx, vperp_idx) / h_initial_vdf_grid.table(vparall_idx, vperp_idx) << '\n';
					}
				}
			}
		}

		//log density history
		{
			std::ofstream ascii_os;
			ascii_os << exceptional_scientific;
			ascii_os.open("./data/density-history.txt");
			for (unsigned idx = 0; idx != vdf_folded_history.size(); ++idx)
				ascii_os << idx * dt << ' ' << vdf_folded_history[idx] << '\n';
		}

		//log mean perp energy
		{
			std::ofstream ascii_os;
			ascii_os << exceptional_scientific;
			ascii_os.open("./data/mean-perp-energy-history.txt");
			for (unsigned idx = 0; idx != vdf_mean_perp_energy_history.size(); ++idx)
				ascii_os << idx * dt << ' ' << vdf_mean_perp_energy_history[idx] << '\n';
		}

		//log result growth rate and amplitude
		{
			table::device_to_host_transfer(growth_rate.growth_rate, h_growth_rate);
			table::device_to_host_transfer(amplitude, h_amplitude);
			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/growth-rate-result.txt");
				for (unsigned idx = 0; idx != h_growth_rate.size; ++idx) {
					ascii_os << h_k_betta(idx) / params.betta_root_c << ' ' << vspace.perp(idx) << ' ' << h_growth_rate(idx) << ' ' << h_amplitude(idx) << '\n';
				}
			}
		}

		//log result velocity distribution function
		{
			table::device_to_host_transfer(diffusion_solver.x_prev, h_result_vdf_grid.table);
			grid::HostGrid<T> h_result_vdf_diff_grid(vspace, vparall_size, vperp_size);
			{
				for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx) {
					for (unsigned vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx)
						h_result_vdf_diff_grid.table(vparall_idx, vperp_idx) = h_result_vdf_grid.table(vparall_idx, vperp_idx) - h_initial_vdf_grid.table(vparall_idx, vperp_idx);
				}
			}

			{
				std::ofstream ascii_os;
				ascii_os << exceptional_scientific;
				ascii_os.open("./data/vdf-result.txt");
				for (unsigned vparall_idx = 0; vparall_idx < vparall_size; vparall_idx += vparall_rate) {
					for (unsigned vperp_idx = 0; vperp_idx < vperp_size; vperp_idx += vperp_rate)
						ascii_os << vspace.perp(vparall_idx) << ' ' << vspace.along(vperp_idx) << ' ' << std::sqrt(2 * std::fabs(vspace.along(vperp_idx))) << ' ' << h_result_vdf_grid.table(vparall_idx, vperp_idx) << ' ' << h_initial_vdf_grid.table(vparall_idx, vperp_idx) << ' ' << h_result_vdf_diff_grid.table(vparall_idx, vperp_idx) << ' ' << h_result_vdf_diff_grid.table(vparall_idx, vperp_idx) / h_initial_vdf_grid.table(vparall_idx, vperp_idx) << '\n';
				}
			}
		}
	}

}/*whfi*/ }/*iki*/
