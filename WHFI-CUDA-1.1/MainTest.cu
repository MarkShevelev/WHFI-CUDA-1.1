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
			auto params = init_parameters(0.95f, 1.0f / 0.95f, 1.f/6.f, -2.f);
			unsigned vparall_size = 512; unsigned vperp_size = 2 * 2048;
			Axis<float> vparall_axis = construct_vparall_axis<float>(params,make_ZFunc<float>(1.e-5f, 15.f), vparall_size, -9.f, -0.96f);
			Axis<float> vperp_axis = { -1e-2f, 2e-2f }; 
			vdf_diffusion<float>(
				params, Space<float>{vparall_axis, vperp_axis}, vparall_size, vperp_size,
				1.0e-5f, 0.f, //amplitude, amplitude time
				2000, 1.f,  //iterations, dt
				false,         //initial vdf export
				false,         //initial core dfc export
				false,         //dfc recalculation
				false, 1000,   //intermidiate growth rate export
				false, 1000,   //intermidiate amplitude export
				false, 1000    //intermidiate vdf export
			);
		}

		if (false) {
			auto params = init_parameters(0.95, 1.0 / 0.95, 1./6., -2.);
			unsigned vparall_size = 512; unsigned vperp_size = 2 * 2048;
			Axis<double> vparall_axis = construct_vparall_axis<double>(params, make_ZFunc<double>(1.e-5, 15.), vparall_size, -9., -0.96);
			Axis<double> vperp_axis = { -1e-2, 2e-2 };
			vdf_diffusion<double>(
				params, Space<double>{vparall_axis, vperp_axis}, vparall_size, vperp_size,
				1.0e-10, 0., //amplitude, amplitude time
				10000, 1.f,  //iterations, dt
				false,         //initial vdf export
				false,         //initial core dfc export
				false,         //dfc recalculation
				false, 1000,   //intermidiate growth rate export
				false, 1000,   //intermidiate amplitude export
				false, 1000    //intermidiate vdf export
			);
		}
	} 
	catch(std::exception const &ex) {
		cout << ex.what() << endl;
	}

	return 0;
}