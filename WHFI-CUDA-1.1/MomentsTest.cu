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
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace iki;
using namespace iki::whfi;
using namespace iki::table;
using namespace iki::grid;

void float_moments_growth_rate_test(PhysicalParameters<float> params, Axis<float> vparall_axis, Axis<float> vperp_axis, unsigned vparall_size, unsigned vperp_size) {
	stringstream params_ss;
	params_ss.precision(2); params_ss.setf(ios::fixed, ios::floatfield);
	params_ss << params.betta_c << '-' << params.bulk_to_alfven_c;

	muVDF<float> mu_vdf(params);
	Space<float> vparall_mu_space = { vparall_axis,vperp_axis };
	auto vdf_grid = calculate_muVDF(params, vparall_mu_space, vparall_size, vperp_size);
	auto vdf_device_table = construct_from(vdf_grid.table);
	auto zero_moment = HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto first_moment = HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto k_betta = HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto dispersion_derive = HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto growth_rate = HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto k_betta_host = HostDataLine<float>(vdf_grid.table.row_count);
	auto dispersion_derive_host = HostDataLine<float>(vdf_grid.table.row_count);

	//k_betta and dispersion_derive calculation
	{
		auto zfunc = make_ZFunc(1.e-5f, 15.f);
		auto wk_solver = ResonantVelocitySolver<float>(zfunc, params);
		auto dr_derive = DispersionRelationOmegaDerivative<float>(zfunc, params);

		for (unsigned idx = 0; idx != vdf_grid.table.row_count; ++idx) {
			auto wk_pair = wk_solver(vparall_axis(idx));
			k_betta_host(idx) = wk_pair.second * params.betta_root_c;
			dispersion_derive_host(idx) = dr_derive(wk_pair.first, wk_pair.second);
		}
		host_to_device_transfer(k_betta_host, k_betta);
		host_to_device_transfer(dispersion_derive_host, dispersion_derive);
	}

	AnalyticalMomentsTable<float> analytical_moments_table(params);
	//zero_moment
	auto g = analytical_moments_table.g(vparall_axis, vdf_grid.table.row_count); 
	//first_moment
	auto G = analytical_moments_table.G(vparall_axis, vdf_grid.table.row_count);

	//zero moment calculation
	{
		dim3 threads(512), blocks((vdf_grid.table.row_count + threads.x - 1) / threads.x);
		cudaError_t cudaStatus;
		whfi::device::zero_moment_kernel <<<blocks, threads>>> (vdf_device_table.table(), vparall_mu_space.along.begin, vparall_mu_space.along.step, zero_moment.line());
		cudaDeviceSynchronize();

		cudaStatus = cudaGetLastError();
		if (cudaSuccess != cudaStatus)
			throw DeviceError(cudaStatus);

		{
			HostGridLine<float> zero_moment_grid(vparall_mu_space.perp, construct_from(zero_moment));
			ofstream ascii_out;
			ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

			ascii_out.open("./data/zero-moment-" + params_ss.str() + "-test.txt");
			ascii_out.precision(7);
			ascii_out.setf(std::ios::scientific, std::ios::floatfield);

			for (unsigned idx = 0; idx != zero_moment_grid.line.size; ++idx) {
				ascii_out << zero_moment_grid.axis(idx) << ' ' << zero_moment_grid.line(idx) << ' ' << g[idx] << ' ' << (zero_moment_grid.line(idx) - g[idx]) << ' ' << (zero_moment_grid.line(idx) - g[idx])/g[idx] << '\n';
			}
		}
	}

	//first moment calculation
	{
		dim3 threads(512), blocks((vdf_grid.table.row_count + threads.x - 1) / threads.x);
		cudaError_t cudaStatus;
		whfi::device::first_moment_kernel <<<blocks, threads>>> (vdf_device_table.table(), vparall_mu_space.along.begin, vparall_mu_space.along.step, first_moment.line());
		cudaDeviceSynchronize();

		cudaStatus = cudaGetLastError();
		if (cudaSuccess != cudaStatus)
			throw DeviceError(cudaStatus);

		{
			HostGridLine<float> first_moment_grid(vparall_mu_space.perp, construct_from(first_moment));
			std::ofstream ascii_out;
			ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

			ascii_out.open("./data/first-moment-" + params_ss.str() + "-test.txt");
			ascii_out.precision(7);
			ascii_out.setf(std::ios::scientific, std::ios::floatfield);

			for (unsigned idx = 0; idx != first_moment_grid.line.size; ++idx) {
				ascii_out << first_moment_grid.axis(idx) << ' ' << first_moment_grid.line(idx) << ' ' << G[idx] << ' ' << (first_moment_grid.line(idx) - G[idx]) << ' ' << (first_moment_grid.line(idx) - G[idx])/G[idx] << '\n';
			}
		}
	}

	{
		dim3 threads(512), blocks((vdf_grid.table.row_count + threads.x - 1) / threads.x);
		cudaError_t cudaStatus;
		whfi::device::growth_rate_kernel <<<blocks, threads>>> (
			zero_moment.line(),
			first_moment.line(),
			k_betta.line(), 
			dispersion_derive.line(),
			vparall_mu_space.perp.step,
			growth_rate.line()
		);
		cudaDeviceSynchronize();

		cudaStatus = cudaGetLastError();
		if (cudaSuccess != cudaStatus)
			throw iki::DeviceError(cudaStatus);

		{
			HostGridLine<float> gr_grid(vparall_mu_space.perp, construct_from(growth_rate));
			std::ofstream ascii_out;
			ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

			ascii_out.open("./data/gamma-" + params_ss.str() + "-test.txt");
			ascii_out.precision(7);
			ascii_out.setf(std::ios::fixed, std::ios::floatfield);

			for (unsigned idx = 1; idx != growth_rate.size - 1; ++idx) {
				ascii_out << vparall_mu_space.perp(idx) << ' ' << k_betta_host(idx) / params.betta_root_c << ' ' << gr_grid.line(idx) << '\n';
			}
		}
	}
}