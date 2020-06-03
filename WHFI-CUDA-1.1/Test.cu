/*#include "TwoDimensionalMultithreadDiffusion.cuh"
#include "Device.h"
#include "DeviceMemory.h"
#include "HostGrid.h"
#include "HostGridIO.h"
#include "HostManagedDeviceTable.cuh"
#include "HostDeviceTransfer.cuh"
#include "HostTableTranspose.h"

#include <iostream>
#include <fstream>

void v_field_init(iki::grid::HostGrid<float> &vdf_grid);
void along_dfc_field_init(iki::grid::HostGrid<float> &along_dfc_grid);
void perp_dfc_field_init(iki::grid::HostGrid<float> &perp_dfc_grid);
void along_mixed_dfc_field_int(iki::grid::HostGrid<float> &vperp_mixed_dfc_grid);
void perp_mixed_dfc_field_int(iki::grid::HostGrid<float> &vparall_mixed_dfc_grid);

int main() {
	using namespace std;
	using namespace iki;
	using namespace table;
	using namespace grid;
	try {
		unsigned vparall_size = 256, vperp_size = 512;
		Space<float> v_space{ Axis<float>{ -15.f, 1.e-3f }, Axis<float>{ 0.f, 1.e-3f } };
		Space<float> v_space_transposed{ Axis<float>{ 0.f, 1.e-3f }, Axis<float>{ -15.f, 1.e-3f } };
		HostGrid<float> vdf_grid(v_space, vparall_size, vperp_size);
		HostGrid<float> vperp_dfc_grid(v_space, vparall_size, vperp_size);
		HostGrid<float> vparall_dfc_grid(v_space_transposed, vperp_size, vparall_size);
		HostGrid<float> vperp_mixed_dfc_grid(v_space, vparall_size, vperp_size);
		HostGrid<float> vparall_mixed_dfc_grid(v_space_transposed, vperp_size, vparall_size);

		v_field_init(vdf_grid);
		along_dfc_field_init(vperp_dfc_grid);
		perp_dfc_field_init(vparall_dfc_grid);
		along_mixed_dfc_field_int(vperp_mixed_dfc_grid);
		perp_mixed_dfc_field_int(vparall_mixed_dfc_grid);

		Device device(0);
		diffusion::TwoDimensionalMultithreadDiffusion<32u, 256u, float> 
			diffusion_solver(
				vdf_grid.table,
				vparall_dfc_grid.table, 1.0f,
				vperp_dfc_grid.table, 1.0f,
				vparall_mixed_dfc_grid.table, vperp_mixed_dfc_grid.table, 1.0f
			);

		for (unsigned iter_cnt = 0; iter_cnt != 1000; ++iter_cnt)
			diffusion_solver.step();

		{
			HostGrid<float> output_grid(v_space, construct_from(diffusion_solver.x_prev));
			ofstream ascii_os;
			ascii_os.exceptions(ios::badbit | ios::failbit);
			ascii_os.precision(7); ascii_os.setf(ios::fixed, ios::floatfield);
			ascii_os.open("./data/two-dimensional-mixed-term-sin-sin-test.txt");
			ascii_os << output_grid;
		}
	}
	catch (exception &ex) {
		cout << ex.what() << endl;
	}

	return 0;
}

void v_field_init(iki::grid::HostGrid<float> &vdf_grid) {
	auto &table = vdf_grid.table;

	for (unsigned prp_idx = 0; prp_idx != table.row_size; ++prp_idx) {
		float val = std::sin(2.f * 3.1415926535f / 128 * prp_idx);
		for (unsigned prl_idx = 0; prl_idx != table.row_count; ++prl_idx)
			table(prl_idx, prp_idx) = val * std::sin(2.f * 3.1415926535f / 128 * prl_idx);
	}

	for (unsigned prp_idx = 0; prp_idx != table.row_size; ++prp_idx)
		table(0, prp_idx) = table(table.row_count - 1, prp_idx) = 0.f;

	for (unsigned prl_idx = 0; prl_idx != table.row_count; ++prl_idx)
		table(prl_idx,0) = table(prl_idx, table.row_size - 1) = 0.f;
}

void along_dfc_field_init(iki::grid::HostGrid<float> &dfc_grid) {
	auto &table = dfc_grid.table;

	for (unsigned row_idx = 0; row_idx != table.row_count; ++row_idx)
		for (unsigned elm_idx = 0; elm_idx != table.row_size; ++elm_idx)
			table(row_idx,elm_idx) = 1.f;
}

void perp_dfc_field_init(iki::grid::HostGrid<float> &dfc_grid) {
	auto &table = dfc_grid.table;

	for (unsigned row_idx = 0; row_idx != table.row_count; ++row_idx)
		for (unsigned elm_idx = 0; elm_idx != table.row_size; ++elm_idx)
			table(row_idx, elm_idx) = 1.f;
}

void along_mixed_dfc_field_int(iki::grid::HostGrid<float> &dfc_grid) {
	auto &table = dfc_grid.table;
	for (unsigned row_idx = 0; row_idx != table.row_count; ++row_idx)
		for (unsigned elm_idx = 0; elm_idx != table.row_size; ++elm_idx)
			table(row_idx, elm_idx) = 1.f;
}

void perp_mixed_dfc_field_int(iki::grid::HostGrid<float> &dfc_grid) {
	auto &table = dfc_grid.table;
	for (unsigned row_idx = 0; row_idx != table.row_count; ++row_idx)
		for (unsigned elm_idx = 0; elm_idx != table.row_size; ++elm_idx)
			table(row_idx, elm_idx) = 1.f;
}*/


#include "ZFunc.h"
#include "ResonantVelocitySolver.h"
#include "HostGrid.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

int main() {
	iki::whfi::PhysicalParamenters<float> params = iki::whfi::init_parameters(0.85f, 1.0f / 0.85f, 0.25f, -1.0f);
	auto wk_solver = iki::whfi::ResonantVelocitySolver<float>(iki::whfi::make_ZFunc(1.e-4f,10.f), params);

	iki::grid::Axis<float> vparall_axis = { -15.0f, 1.e-2 };

	try {
		std::ofstream ascii_out;
		ascii_out.exceptions(std::ios::badbit | std::ios::failbit);

		ascii_out.open("./data/vres-omega-k-test.txt");
		ascii_out.precision(7); ascii_out.setf(std::ios::fixed, std::ios::floatfield);
		for (unsigned idx = 0; true; ++idx) {
			auto v_res = vparall_axis(idx);
			if (v_res > -0.9f) break;
			auto wk_pair = wk_solver(v_res);
			ascii_out << v_res << ' ' << wk_pair.first << ' ' << wk_pair.second << std::endl;
		}
	}
	catch (std::exception const &ex) {
		std::cout << ex.what() << std::endl;
	}
	
	return 0;
}