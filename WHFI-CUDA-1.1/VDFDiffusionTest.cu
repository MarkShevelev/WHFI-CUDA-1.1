#include "PhysicalParameters.h"
#include "HostManagedDeviceTable.cuh"
#include "HostManagedDeviceDataLine.cuh"
#include "HostTable.h"
#include "HostDataLine.h"
#include "HostGrid.h"
#include "HostTableTranspose.h"
#include "HostGridIO.h"
#include "VelocityDistributionFunction.h"
#include "initial_diffusion_coefficients_calculation.cuh"
#include "Device.h"
#include "TwoDimensionalMultithreadDiffusion.cuh"
#include "host_math_helper.h"
#include "GrowthRateCalculation.cuh"
#include "diffusion_coefficients_recalculation_kernel.cuh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace iki;
using namespace iki::whfi;
using namespace iki::table;
using namespace iki::grid;

void float_vdf_diffusion_test(PhysicalParameters<float> params, Axis<float> vparall_axis, Axis<float> vperp_axis, unsigned vparall_size, unsigned vperp_size, float dt, unsigned iteration) {

	Space<float> vspace = { vparall_axis,vperp_axis };
	Space<float> vspace_transposed = { vperp_axis ,vparall_axis };
	auto h_vdf_grid = calculate_muVDF(params, vspace, vparall_size, vperp_size);
	
	HostDataLine<float> h_amplitude_spectrum(vparall_size);
	HostTable<float> h_dfc_vperp_vperp(vparall_size, vperp_size);
	HostTable<float> h_dfc_vparall_vparall(vperp_size, vparall_size); //transposed
	HostTable<float> h_dfc_vparall_vperp(vparall_size, vperp_size);
	HostTable<float> h_dfc_vperp_vparall(vperp_size, vparall_size);   //transposed
	HostDataLine<float> h_k_betta(vparall_size);
	HostDataLine<float> h_dispersion_derive(vparall_size);
	initial_diffusion_coefficients_calculation<float>(
		params, 
		vparall_axis, vperp_axis, 
		vparall_size, vperp_size,
		h_dfc_vperp_vperp, h_dfc_vparall_vperp,
		h_dfc_vparall_vparall, h_dfc_vperp_vparall, //transposed
		h_k_betta, h_dispersion_derive
	);
	std::cout << "diffusion coefficients calculated" << std::endl;

	if (false) {
		HostGrid<float> h_dfc_vperp_vperp_grid(vspace, HostTable<float>(h_dfc_vperp_vperp));
		HostGrid<float> h_dfc_vparall_vparall_grid(vspace_transposed, HostTable<float>(h_dfc_vparall_vparall));
		HostGrid<float> h_dfc_vparall_vperp_grid(vspace, HostTable<float>(h_dfc_vparall_vperp));
		HostGrid<float> h_dfc_vperp_vparall_grid(vspace_transposed, HostTable<float>(h_dfc_vperp_vparall));
		{
			std::ofstream ascii_os;
			ascii_os.exceptions(ios::failbit | ios::badbit);
			ascii_os.open("./data/whfi-diffusion-coeff-vperp-vperp.txt");
			ascii_os << h_dfc_vperp_vperp_grid;
		}

		{
			std::ofstream ascii_os;
			ascii_os.exceptions(ios::failbit | ios::badbit);
			ascii_os.open("./data/whfi-diffusion-coeff-vparall-vparall.txt");
			ascii_os << h_dfc_vparall_vparall_grid;
		}

		{
			std::ofstream ascii_os;
			ascii_os.exceptions(ios::failbit | ios::badbit);
			ascii_os.open("./data/whfi-diffusion-coeff-vparall-vperp.txt");
			ascii_os << h_dfc_vparall_vperp_grid;
		}

		{
			std::ofstream ascii_os;
			ascii_os.exceptions(ios::failbit | ios::badbit);
			ascii_os.open("./data/whfi-diffusion-coeff-vperp-vparall.txt");
			ascii_os << h_dfc_vperp_vparall_grid;
		}

		std::cout << "diffusin coefficients exported" << std::endl;
		return;
	}

	Device device(0);

	table::HostManagedDeviceTable<float> 
		core_dfc_vperp_vperp(table::construct_from(h_dfc_vperp_vperp)),
		core_dfc_vparall_vparall(table::construct_from(h_dfc_vparall_vparall)), //transposed
		core_dfc_vparall_vperp(table::construct_from(h_dfc_vparall_vperp)),
		core_dfc_vperp_vparall(table::construct_from(h_dfc_vperp_vparall))      //transposed
	;
	table::HostManagedDeviceDataLine<float> amplitude_spectrum(table::construct_from(h_amplitude_spectrum));

	float rvperp = dt / math::pow<2u>(vspace.along.step), rvparall = dt / math::pow<2u>(vspace.perp.step), rmixed = std::sqrt(rvperp*rvparall);
	diffusion::TwoDimensionalMultithreadDiffusion<32u, 256u, float>
		diffusion_solver(
			h_vdf_grid.table,
			h_dfc_vparall_vparall, rvperp,
			h_dfc_vperp_vperp, rvparall,
			h_dfc_vperp_vparall, h_dfc_vparall_vperp, rmixed
		);

	GrowthRateCalculation<float> growth_rate(vspace, h_k_betta, h_dispersion_derive);
	if (true) {
		growth_rate.recalculate(diffusion_solver.x_prev);
		HostDataLine<float> h_growth_rate(construct_from(growth_rate.growth_rate));
		{
			ofstream ascii_out;
			ascii_out.exceptions(ios::failbit | ios::badbit);
			ascii_out.precision(7); ascii_out.setf(ios::scientific, ios::floatfield);
			ascii_out.open("./data/growth-rate-initial-test.txt");
			for (unsigned idx = 0; idx != vparall_size; ++idx) {
				ascii_out << h_k_betta(idx) << ' ' << h_dispersion_derive(idx) << ' ' << h_k_betta(idx) / params.betta_root_c << ' ' << h_growth_rate(idx) << '\n';
			}
		}

		for (unsigned idx = 0; idx != vparall_size; ++idx) {
			h_amplitude_spectrum(idx) = 1.0e-5f * std::exp(h_growth_rate(idx) * dt * 100000);
		}
		
	}

	//diffusion coefficients adjusment
	{
		table::HostManagedDeviceDataLine<float> amplitude_spectrum(table::construct_from(h_amplitude_spectrum));
		whfi::device::diffusion_coefficients_recalculation_kernel<<<vparall_size/512,512>>> (
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
	}

	for (unsigned iter_cnt = 0; iter_cnt != iteration; ++iter_cnt)
		diffusion_solver.step();
	std::cout << "iterations made" << std::endl;

	{
		HostGrid<float> h_result_vdf_grid(vspace, construct_from(diffusion_solver.x_prev));
		HostGrid<float> h_diff_vdf_grid(vspace, vparall_size, vperp_size);

		for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx)
			for (unsigned vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx)
				h_diff_vdf_grid.table(vparall_idx, vperp_idx) = h_result_vdf_grid.table(vparall_idx, vperp_idx) / h_vdf_grid.table(vparall_idx, vperp_idx) - 1.0f;//1.0f - h_vdf_grid.table(vparall_idx, vperp_idx) / h_result_vdf_grid.table(vparall_idx, vperp_idx);
		{
			ofstream ascii_os;
			ascii_os.exceptions(ios::badbit | ios::failbit);
			ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
			ascii_os.open("./data/two-dimensional-mixed-term-result-vdf-test.txt");
			ascii_os << h_result_vdf_grid;
		}
		
		//HostGrid<float> h_diff_vdf_grid_transposed(vspace_transposed, vperp_size, vparall_size);
		//transpose(h_diff_vdf_grid.table, h_diff_vdf_grid_transposed.table);
		{
			ofstream ascii_os;
			ascii_os.exceptions(ios::badbit | ios::failbit);
			ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
			ascii_os.open("./data/two-dimensional-mixed-term-diff-vdf-test.txt");
			ascii_os << h_diff_vdf_grid;
		}

		if (true) {
			growth_rate.recalculate(diffusion_solver.x_prev);
			HostGridLine<float> growth_rate_grid(vspace.perp, construct_from(growth_rate.growth_rate));
			{
				ofstream ascii_out;
				ascii_out.exceptions(ios::failbit | ios::badbit);
				ascii_out.precision(7); ascii_out.setf(ios::scientific, ios::floatfield);
				ascii_out.open("./data/growth-rate-result-test.txt");
				for (unsigned idx = 0; idx != vparall_size; ++idx) {
					ascii_out << h_k_betta(idx) << ' ' << h_dispersion_derive(idx) << ' ' << h_k_betta(idx) / params.betta_root_c << ' ' << growth_rate_grid.line(idx) << '\n';
				}
			}
		}
	}
	std::cout << "data exported" << std::endl;
}