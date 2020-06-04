#include "DeviceError.h"
#include "HostGrid.h"
#include "HostGridIO.h"
#include "HostDeviceTransfer.cuh"
#include "HostManagedDeviceDataLine.cuh"
#include "VelocityDistributionFunction.h"
#include "zero_moment_kernel.cuh"
#include "first_moment_kernel.cuh"
#include "AnalyticalMoments.h"

#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <stdexcept>

int main() {
	

	iki::whfi::PhysicalParameters<float> params = iki::whfi::init_parameters(0.85f, 1.0f / 0.85f, 0.25f, -9.0f);
	iki::whfi::muVDF<float> mu_vdf(params);

	iki::grid::Space<float> vparall_mu_space = {
		iki::grid::Axis<float>{-16.5f, 1.5e-2f},
		iki::grid::Axis<float>{0.f, 1.5e-2f}
	};
	auto vdf_grid = iki::whfi::calculate_muVDF(params, vparall_mu_space, 1024, 1024);
	auto vdf_device_table = iki::table::construct_from(vdf_grid.table);
	auto zero_moment = iki::table::HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);
	auto first_moment = iki::table::HostManagedDeviceDataLine<float>(vdf_grid.table.row_count);

	try {
		{
			dim3 threads(512), blocks(vdf_grid.table.row_count / threads.x);
			cudaError_t cudaStatus;
			iki::whfi::device::zero_moment_kernel <<<blocks, threads>>> (vdf_device_table.table(), vparall_mu_space.along.begin, vparall_mu_space.along.step, zero_moment.line());
			cudaDeviceSynchronize();

			cudaStatus = cudaGetLastError();
			if (cudaSuccess != cudaStatus)
				throw iki::DeviceError(cudaStatus);
		}

		{
			dim3 threads(512), blocks(vdf_grid.table.row_count / threads.x);
			cudaError_t cudaStatus;
			iki::whfi::device::first_moment_kernel <<<blocks, threads>>> (vdf_device_table.table(), vparall_mu_space.along.begin, vparall_mu_space.along.step, first_moment.line());
			cudaDeviceSynchronize();

			cudaStatus = cudaGetLastError();
			if (cudaSuccess != cudaStatus)
				throw iki::DeviceError(cudaStatus);
		}


		{
			iki::grid::HostGridLine<float> zero_moment_grid(vparall_mu_space.perp, iki::table::construct_from(zero_moment));
			std::ofstream ascii_out;
			ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

			ascii_out.open("./data/device-zero-moment-test.txt");
			ascii_out.precision(7);
			ascii_out.setf(std::ios::fixed, std::ios::floatfield);

			ascii_out << zero_moment_grid;
		}

		{
			iki::grid::HostGridLine<float> first_moment_grid(vparall_mu_space.perp, iki::table::construct_from(first_moment));
			std::ofstream ascii_out;
			ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

			ascii_out.open("./data/device-first-moment-test.txt");
			ascii_out.precision(7);
			ascii_out.setf(std::ios::fixed, std::ios::floatfield);

			ascii_out << first_moment_grid;
		}

		{
			iki::whfi::AnalyticalMoments<float> analytical_moments(params);
			auto g = analytical_moments.g(vparall_mu_space.perp.begin, vparall_mu_space.perp.step, 1024);
			auto G = analytical_moments.G(vparall_mu_space.perp.begin, vparall_mu_space.perp.step, 1024);

			{
				std::ofstream ascii_out;
				ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

				ascii_out.open("./data/analytical-zero-moment.txt");
				ascii_out.precision(7);
				ascii_out.setf(std::ios::fixed, std::ios::floatfield);

				for (auto x : g) {
					ascii_out << x << '\n';
				}
			}

			{
				std::ofstream ascii_out;
				ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

				ascii_out.open("./data/analytical-first-moment.txt");
				ascii_out.precision(7);
				ascii_out.setf(std::ios::fixed, std::ios::floatfield);

				for (auto x : G) {
					ascii_out << x << '\n';
				}
			}
		}
	}
	catch (std::exception &ex) {
		std::cout << ex.what() << std::endl;
	}

	return 0;
}