#include "TwoDimensionalMultithreadDiffusion.cuh"
#include "Device.h"
#include "DeviceMemory.h"
#include "DataTable.h"
#include "UniformGrid.h"
#include "UniformGridIO.h"
#include "DataTableDeviceHelper.h"

#include <iostream>
#include <fstream>

void v_field_init(iki::grid::UniformGrid<float, 2u, 1u> &v_field);
void along_dfc_field_init(iki::grid::UniformGrid<float, 2u, 1u> &dfc_field);
void perp_dfc_field_init(iki::grid::UniformGrid<float, 2u, 1u> &dfc_field);

int main() {
	using namespace std;
	using namespace iki;
	using namespace table;
	using namespace grid;
	try {
		Bounds<2u> v_field_size = { 256, 512 };
		UniformSpace<float, 2u> v_space = { Axis<float>{0., 1.e-3f}, Axis<float>{-15.f, 1.e-3f} };

		Bounds<2u> transposed_field_size = { v_field_size[1], v_field_size[0] };
		UniformSpace<float, 2u> transposed_space = { Axis<float>{-15.f, 1.e-3f}, Axis<float>{0., 1.e-3f} };
		
		UniformGrid<float, 2u, 1u> v_field(v_space,v_field_size);
		v_field_init(v_field);

		UniformGrid<float, 2u, 1u> along_dfc_field(v_space, v_field_size);
		along_dfc_field_init(along_dfc_field);

		UniformGrid<float, 2u, 1u> perp_dfc_field(transposed_space,transposed_field_size);
		perp_dfc_field_init(perp_dfc_field);

		Device device(0);

		unsigned const row_size = v_field_size[1], row_count = v_field_size[0], field_size = row_size*row_count;
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
		
		device::to_device(x_prev, v_field.table);
		device::to_device(x_next, v_field.table);
		device::to_device(along_dfc, along_dfc_field.table);
		device::to_device(perp_dfc, perp_dfc_field.table);

		diffusion::TwoDimensionalMultithreadDiffusion<32u, 256u, float> diffusion_solver(row_count, row_size, a, b, c, d, x_prev, x_next, x_tmp, along_dfc, 1.0f, perp_dfc, 1.0f);
		for (unsigned iter_cnt = 0; iter_cnt != 1000; ++iter_cnt)
			diffusion_solver.step();

		device::from_device(v_field.table, diffusion_solver.x_prev);
		{
			ofstream ascii_os;
			ascii_os.exceptions(ios::badbit | ios::failbit);
			ascii_os.precision(7); ascii_os.setf(ios::fixed, ios::floatfield);
			ascii_os.open("./data/one-dimensional-sin-test.txt");
			ascii_os << transposed_grid(v_field);
		}

	}
	catch (exception &ex) {
		cout << ex.what() << endl;
	}

	return 0;
}

void v_field_init(iki::grid::UniformGrid<float, 2u, 1u> &v_field) {
	auto &table = v_field.table;
	auto &bounds = table.get_bounds();

	for (unsigned elm_idx = 0; elm_idx != bounds[1]; ++elm_idx) {
		float val = std::sin(2.f * 3.1415926535f / 128 * elm_idx);
		for (unsigned row_idx = 0; row_idx != bounds[0]; ++row_idx)
			*(table[row_idx + elm_idx * bounds[0]].begin()) = val;
	}

}

void along_dfc_field_init(iki::grid::UniformGrid<float, 2u, 1u> &dfc_field) {
	auto &table = dfc_field.table;
	auto &bounds = table.get_bounds();

	for (unsigned row_idx = 0; row_idx != bounds[0]; ++row_idx)
		for (unsigned elm_idx = 0; elm_idx != bounds[1]; ++elm_idx)
			*(table[row_idx + elm_idx * bounds[0]].begin()) = 1.f;
}

void perp_dfc_field_init(iki::grid::UniformGrid<float, 2u, 1u> &dfc_field) {
	auto &table = dfc_field.table;
	auto &bounds = table.get_bounds();

	for (unsigned row_idx = 1; row_idx != bounds[0] - 2; ++row_idx)
		for (unsigned elm_idx = 1; elm_idx != bounds[1] - 2; ++elm_idx)
			*(table[row_idx + elm_idx * bounds[0]].begin()) = 1.f;
}