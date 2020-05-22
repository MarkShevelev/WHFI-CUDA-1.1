#include "TwoDimensionalMultithreadDiffusion.cuh"
#include "Device.h"
#include "DeviceMemory.h"
#include "DataTable.h"
#include "UniformGrid.h"
#include "UniformGridIO.h"
#include "DataTableDeviceHelper.h"
#include "HostGrid.h"
#include "HostGridIO.h"

#include <iostream>
#include <fstream>

void v_field_init(iki::grid::test::HostGrid<float> &vdf_grid);
void along_dfc_field_init(iki::grid::test::HostGrid<float> &dfc_grid);
void perp_dfc_field_init(iki::grid::test::HostGrid<float> &dfc_grid);

int main() {
	using namespace std;
	using namespace iki;
	using namespace table;
	using namespace grid;
	try {
		unsigned vparall_size = 256, vperp_size = 512;
		test::Space<float> v_space{ test::Axis<float>{ -15.f, 1.e-3f }, test::Axis<float>{ 0.f, 1.e-3f } };
		test::Space<float> v_space_transposed{ test::Axis<float>{ 0.f, 1.e-3f }, test::Axis<float>{ -15.f, 1.e-3f } };
		test::HostGrid<float> vdf_grid(v_space, vparall_size, vperp_size);
		test::HostGrid<float> vperp_dfc_grid(v_space, vparall_size, vperp_size);
		test::HostGrid<float> vparall_dfc_grid(v_space_transposed, vperp_size, vparall_size);

		v_field_init(vdf_grid);
		along_dfc_field_init(vperp_dfc_grid);
		perp_dfc_field_init(vparall_dfc_grid);

		Device device(0);

		unsigned const row_count = vparall_size, row_size = vperp_size, field_size = row_size*row_count;
		DeviceMemory dev_memory(field_size * 9 * sizeof(float));
		float *x_prev = (float *)dev_memory;
		float *x_next = x_prev + field_size;
		float *x_tmp = x_next + field_size;
		float *along_dfc = x_tmp + field_size;
		float *perp_dfc = along_dfc + field_size;
		float *a = perp_dfc + field_size;
		float *b = a + field_size;
		float *c = b + field_size;
		float *d = c + field_size;

		cudaMemcpy(x_prev, vdf_grid.table.hData.data(), field_size * sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(x_next, vdf_grid.table.hData.data(), field_size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(along_dfc, vperp_dfc_grid.table.hData.data(), field_size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(perp_dfc, vparall_dfc_grid.table.hData.data(), field_size * sizeof(float), cudaMemcpyHostToDevice);

		diffusion::TwoDimensionalMultithreadDiffusion<32u, 256u, float> diffusion_solver(row_count, row_size, a, b, c, d, x_prev, x_next, x_tmp, along_dfc, 1.0f, perp_dfc, 1.0f);
		for (unsigned iter_cnt = 0; iter_cnt != 1000; ++iter_cnt)
			diffusion_solver.step();

		cudaMemcpy(vdf_grid.table.hData.data(), x_prev, field_size * sizeof(float), cudaMemcpyDeviceToHost);
		{
			ofstream ascii_os;
			ascii_os.exceptions(ios::badbit | ios::failbit);
			ascii_os.precision(7); ascii_os.setf(ios::fixed, ios::floatfield);
			ascii_os.open("./data/one-dimensional-sin-test.txt");
			ascii_os << vdf_grid;
		}

	}
	catch (exception &ex) {
		cout << ex.what() << endl;
	}

	return 0;
}

void v_field_init(iki::grid::test::HostGrid<float> &vdf_grid) {
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

void along_dfc_field_init(iki::grid::test::HostGrid<float> &dfc_grid) {
	auto &table = dfc_grid.table;

	for (unsigned row_idx = 0; row_idx != table.row_count; ++row_idx)
		for (unsigned elm_idx = 0; elm_idx != table.row_size; ++elm_idx)
			table(row_idx,elm_idx) = 1.f;
}

void perp_dfc_field_init(iki::grid::test::HostGrid<float> &dfc_grid) {
	auto &table = dfc_grid.table;

	for (unsigned row_idx = 0; row_idx != table.row_count; ++row_idx)
		for (unsigned elm_idx = 0; elm_idx != table.row_size; ++elm_idx)
			table(row_idx, elm_idx) = 1.f;
}