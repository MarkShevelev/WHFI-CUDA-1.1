#include "PhysicalParameters.h"
#include "HostGrid.h"

#include <iostream>
#include <exception>

using namespace std;
using namespace iki;
using namespace iki::whfi;
using namespace iki::table;
using namespace iki::grid;

void float_diffusion_coefficient_test(PhysicalParameters<float> params, Axis<float> vparall_axis, Axis<float> vperp_axis, unsigned vparall_size, unsigned vperp_size);

void float_vdf_diffusion_test(PhysicalParameters<float> params, Axis<float> vparall_axis, Axis<float> vperp_axis, unsigned vparall_size, unsigned vperp_size, float dt, unsigned iter);

int main() {
	try {
		auto params = init_parameters(0.85f, 1.0f / 0.85f, 0.25f, -9.f);
		Axis<float> vparall_axis = { -16.3f, 3e-2f }; unsigned vparall_size = 512;
		Axis<float> vperp_axis = { -3.e-2f, 3.e-2f }; unsigned vperp_size = 1024;
		float_vdf_diffusion_test(params, vparall_axis, vperp_axis, vparall_size, vperp_size, 1.e-3f, 1000);
	} 
	catch(std::exception const &ex) {
		cout << ex.what() << endl;
	}

	return 0;
}