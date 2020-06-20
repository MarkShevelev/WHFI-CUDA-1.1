#include "PhysicalParameters.h"
#include "HostGrid.h"
#include "template_vdf_diffusion.cuh"
#include "ZFunc.h"
#include "construct_vparall_axis.h"

#include <iostream>
#include <exception>

using namespace std;
using namespace iki;
using namespace iki::whfi;
using namespace iki::table;
using namespace iki::grid;

void float_automated_vparall_axis_test(PhysicalParameters<float> params, unsigned vparall_size, float begin, float end);

int main() {
	try {
		if (true) {
			auto params = init_parameters(0.85f, 1.0f / 0.85f, 0.25f, -9.f);
			unsigned vparall_size = 512; unsigned vperp_size = 1024;
			Axis<float> vparall_axis = construct_vparall_axis<float>(params,make_ZFunc<float>(1.e-5f, 15.f), vparall_size, -7.f, -0.96f);
			Axis<float> vperp_axis = { -3e-2f, 3e-2f }; 
			vdf_diffusion<float>(
				params, Space<float>{vparall_axis, vperp_axis}, vparall_size, vperp_size,
				1.0e-6f, 0.f, //amplitude, amplitude time
				20000, 0.1f,  //iterations, dt
				true,
				true,
				true,
				true,
				true);
		}

		if (false) {
			auto params = init_parameters(0.85, 1.0 / 0.85, 0.25, -9.);
			Axis<double> vparall_axis = { -16.3, 3e-2 }; unsigned vparall_size = 512;
			Axis<double> vperp_axis = { -3e-2, 3e-2 }; unsigned vperp_size = 1024;
			vdf_diffusion<double>(
				params, Space<double>{vparall_axis, vperp_axis}, vparall_size, vperp_size,
				1.0e-5, 0., //amplitude, amplitude time
				10000, 0.1,        //iterations, dt
				true,
				true,
				true,
				true,
				true);
		}
	} 
	catch(std::exception const &ex) {
		cout << ex.what() << endl;
	}

	return 0;
}