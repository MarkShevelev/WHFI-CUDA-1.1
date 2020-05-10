#include "OneDimensionalMultithreadDiffusion.cuh"
#include "Device.h"
#include "DeviceMemory.h"
#include "DataTable.h"
#include "UniformGrid.h"
#include "DataTableDeviceHelper.h"

#include <iostream>

void v_field_init(iki::grid::UniformGrid<float, 2u, 1u> &v_field);

int main() {
	using namespace std;
	using namespace iki;
	using namespace table;
	using namespace grid;
	try {
		Bounds<2u> v_field_size = { 1024, 2048 };
		UniformSpace<float, 2u> v_space = { 1.e-3, 1.e-3 };
		UniformGrid<float, 2u, 1u> v_field;
		{
			v_field.space = v_space;
			v_field.table.set_bounds(v_field_size);
		}
		v_field_init(v_field);

		Device device(0);

		unsigned const row_size = v_field_size[1], row_count = v_field_size[0], field_size = row_size*row_count;
		DeviceMemory dev_memory(field_size * 7 * sizeof(float));
		float *x_next = (float *)dev_memory;
		float *x_prev = x_next + field_size;
		float *dfc = x_prev + field_size;
		float *a = dfc + field_size;
		float *b = a + field_size;
		float *c = b + field_size;
		float *d = c + field_size;
		
		device::to_device(x_prev, v_field.table);
		device::to_device(x_next, v_field.table);

		diffusion::OneDimensionalMultithreadDiffusion<32u,512u,float> diffusion_solver(1.f, row_size, row_count, x_next, x_prev, dfc, a, b, c, d);

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
		float val = std::sin(2. * 3.1415926535 / 64 * elm_idx);
		for (unsigned row_idx = 0; row_idx != bounds[0]; ++row_idx)
			*(table[row_idx + elm_idx * bounds[0]].begin()) = val;
	}

}