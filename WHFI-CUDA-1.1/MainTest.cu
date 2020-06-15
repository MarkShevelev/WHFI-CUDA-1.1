#include "PhysicalParameters.h"
#include "HostGrid.h"
#include "template_vdf_diffusion.cuh"

#include <iostream>
#include <exception>

using namespace std;
using namespace iki;
using namespace iki::whfi;
using namespace iki::table;
using namespace iki::grid;

int main() {
	try {
		auto params = init_parameters(0.85f, 1.0f / 0.85f, 0.25f, -9.f);
		Axis<float> vparall_axis = { -16.3f, 3e-2f }; unsigned vparall_size = 512;
		Axis<float> vperp_axis = { -3e-2f, 3e-2f }; unsigned vperp_size = 1024;
		vdf_diffusion<float>(
			params, Space<float>{vparall_axis, vperp_axis}, vparall_size, vperp_size,
			1.0e-5f, 0.f, //amplitude, amplitude time
			10000, 0.1f,        //iterations, dt
			true,
			true,
			true,
			true,
			true);
	} 
	catch(std::exception const &ex) {
		cout << ex.what() << endl;
	}

	return 0;
}