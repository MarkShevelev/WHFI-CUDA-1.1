#include "OneDimensionalMultithreadDiffusion.cuh"
#include "Device.h"
#include "DeviceMemory.h"

#include <iostream>

int main() {
	using namespace std;
	using namespace iki;
	try {
		Device device(0);

		unsigned const row_size = 1024, row_count = 2048, field_size = row_size*row_count;
		DeviceMemory dev_memory(field_size * 7 * sizeof(float));
		float *x_next = (float *)dev_memory;
		float *x_prev = x_next + field_size;
		float *dfc = x_prev + field_size;
		float *a = dfc + field_size;
		float *b = a + field_size;
		float *c = b + field_size;
		float *d = c + field_size;

		diffusion::OneDimensionalMultithreadDiffusion<32u,512u,float> diffusion_solver(1.f, row_size, row_count, x_next, x_prev, dfc, a, b, c, d);
	}
	catch (exception &ex) {
		cout << ex.what() << endl;
	}

	return 0;
}