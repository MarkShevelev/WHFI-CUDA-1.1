#include "DeviceError.h"
#include "HostGrid.h"
#include "HostGridIO.h"
#include "HostDeviceTransfer.cuh"
#include "HostManagedDeviceDataLine.cuh"
#include "HostDataLine.h"
#include "VelocityDistributionFunction.h"
#include "zero_moment_kernel.cuh"
#include "first_moment_kernel.cuh"
#include "AnalyticalMoments.h"
#include "ZFunc.h"
#include "ResonantVelocitySolver.h"
#include "DispersionRelationDerivative.h"
#include "growth_rate_kernel.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <stdexcept>

int main() {
	iki::whfi::PhysicalParameters<float> params = iki::whfi::init_parameters(0.85f, 1.0f / 0.85f, 0.25f, -9.0f);
	iki::whfi::muVDF<float> mu_vdf(params);

	iki::grid::Space<float> vparall_mu_space = {
		iki::grid::Axis<float>{-16.5f, 3.e-2f},
		iki::grid::Axis<float>{0.f, 3.e-2f}
	};
	auto vdf_grid = iki::whfi::calculate_muVDF(params, vparall_mu_space, 512, 1024);
	auto vdf_device_table = iki::table::construct_from(vdf_grid.table);
	auto zero_moment = iki::table::HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto first_moment = iki::table::HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto k_betta = iki::table::HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto dispersion_derive = iki::table::HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto gamma = iki::table::HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto k_betta_host = iki::table::HostDataLine<float>(vdf_grid.table.row_count);
	auto dispersion_derive_host = iki::table::HostDataLine<float>(vdf_grid.table.row_count);

	try {
		//k_betta and dispersion_derive calculation
		{
			auto zfunc = iki::whfi::make_ZFunc(1.e-5f, 15.f);
			auto wk_solver = iki::whfi::ResonantVelocitySolver<float>(zfunc, params);
			auto dr_derive = iki::whfi::DispersionRelationOmegaDerivative<float>(zfunc, params);

			auto const &vparall_axis = vparall_mu_space.perp;
			for (unsigned idx = 0; idx != vdf_grid.table.row_count; ++idx) {
				auto wk_pair = wk_solver(vparall_axis(idx));
				k_betta_host(idx) = wk_pair.second * params.betta_root_c;
				dispersion_derive_host(idx) = dr_derive(wk_pair.first, wk_pair.second);
			}
			iki::table::host_to_device_transfer(k_betta_host, k_betta);
			iki::table::host_to_device_transfer(dispersion_derive_host, dispersion_derive);
		}

		iki::whfi::AnalyticalMoments<float> analytical_moments(params);
		auto g = analytical_moments.g(vparall_mu_space.perp.begin, vparall_mu_space.perp.step, vdf_grid.table.row_count); //zero_moment
		auto G = analytical_moments.G(vparall_mu_space.perp.begin, vparall_mu_space.perp.step, vdf_grid.table.row_count);//first_moment

		//zero moment calculation
		{
			dim3 threads(512), blocks(vdf_grid.table.row_count / threads.x);
			cudaError_t cudaStatus;
			iki::whfi::device::zero_moment_kernel <<<blocks, threads>>> (vdf_device_table.table(), vparall_mu_space.along.begin, vparall_mu_space.along.step, zero_moment.line());
			cudaDeviceSynchronize();

			cudaStatus = cudaGetLastError();
			if (cudaSuccess != cudaStatus)
				throw iki::DeviceError(cudaStatus);

			{
				iki::grid::HostGridLine<float> zero_moment_grid(vparall_mu_space.perp, iki::table::construct_from(zero_moment));
				std::ofstream ascii_out;
				ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

				ascii_out.open("./data/zero-moment-test.txt");
				ascii_out.precision(7);
				ascii_out.setf(std::ios::scientific, std::ios::floatfield);

				for (unsigned idx = 0; idx != zero_moment_grid.line.size; ++idx) {
					ascii_out << zero_moment_grid.axis(idx) << ' ' << zero_moment_grid.line(idx) << ' ' << g[idx] << ' ' << (zero_moment_grid.line(idx) - g[idx]) << ' ' << (zero_moment_grid.line(idx) - g[idx])/g[idx] << '\n';
				}
			}
		}

		//first moment calculation
		{
			dim3 threads(512), blocks(vdf_grid.table.row_count / threads.x);
			cudaError_t cudaStatus;
			iki::whfi::device::first_moment_kernel <<<blocks, threads>>> (vdf_device_table.table(), vparall_mu_space.along.begin, vparall_mu_space.along.step, first_moment.line());
			cudaDeviceSynchronize();

			cudaStatus = cudaGetLastError();
			if (cudaSuccess != cudaStatus)
				throw iki::DeviceError(cudaStatus);

			{
				iki::grid::HostGridLine<float> first_moment_grid(vparall_mu_space.perp, iki::table::construct_from(first_moment));
				std::ofstream ascii_out;
				ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

				ascii_out.open("./data/first-moment-test.txt");
				ascii_out.precision(7);
				ascii_out.setf(std::ios::scientific, std::ios::floatfield);

				for (unsigned idx = 0; idx != first_moment_grid.line.size; ++idx) {
					ascii_out << first_moment_grid.axis(idx) << ' ' << first_moment_grid.line(idx) << ' ' << G[idx] << ' ' << (first_moment_grid.line(idx) - G[idx]) << ' ' << (first_moment_grid.line(idx) - G[idx])/G[idx] << '\n';
				}
			}
		}

		{
			dim3 threads(512), blocks(vdf_grid.table.row_count / threads.x);
			cudaError_t cudaStatus;
			iki::whfi::device::growth_rate_kernel <<<blocks, threads>>> (
				zero_moment.line(),
				first_moment.line(),
				k_betta.line(), 
				dispersion_derive.line(),
				vparall_mu_space.perp.step,
				gamma.line()
			);
			cudaDeviceSynchronize();

			cudaStatus = cudaGetLastError();
			if (cudaSuccess != cudaStatus)
				throw iki::DeviceError(cudaStatus);

			{
				iki::grid::HostGridLine<float> gamma_grid(vparall_mu_space.perp, iki::table::construct_from(gamma));
				std::ofstream ascii_out;
				ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

				ascii_out.open("./data/gamma-test.txt");
				ascii_out.precision(7);
				ascii_out.setf(std::ios::fixed, std::ios::floatfield);

				for (unsigned idx = 1; idx != gamma.size - 1; ++idx) {
					ascii_out << vparall_mu_space.perp(idx) << ' ' << k_betta_host(idx) / params.betta_root_c << ' ' << gamma_grid.line(idx) << '\n';
				}
			}
		}
	}
	catch (std::exception &ex) {
		std::cout << ex.what() << std::endl;
	}

	return 0;
}