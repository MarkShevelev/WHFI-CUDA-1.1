#include "PhysicalParameters.h"
#include "HostGrid.h"
#include "template_vdf_diffusion.cuh"
#include "ZFunc.h"
#include "construct_vparall_axis.h"

#include <iostream>
#include <fstream>
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
			unsigned vparall_size = 512; unsigned vperp_size = 3072;
			Axis<float> vparall_axis = construct_vparall_axis<float>(params,make_ZFunc<float>(1.e-5f, 15.f), vparall_size, -12.f, -0.96f);
			Axis<float> vperp_axis = { -0.75e-2f, 1.5e-2f };

			//analytical growth rate
			if (false) {
				auto zfunc = make_ZFunc<float>(1.e-5f, 15.f);
				auto moments = AnalyticalMoments<float>(params);
				auto vres_solver = ResonantVelocitySolver<float>(zfunc, params);
				auto dDdw = DispersionRelationOmegaDerivative<float>(zfunc, params);

				{
					ofstream ascii_os;
					ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
					ascii_os.exceptions(ios::badbit | ios::failbit);
					ascii_os.open("./data/growth-rate-analytical.txt");
					for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx) {
						auto vparall = vparall_axis(vparall_idx);
						auto wk_pair = vres_solver(vparall);
						auto k_betta = wk_pair.second * params.betta_root_c;
						auto growth_rate = -float(1.25331414) * (moments.GDerive(vparall) - moments.g(vparall) / k_betta) / dDdw(wk_pair.first, wk_pair.second);
						ascii_os << wk_pair.second << ' ' << vparall << ' ' << growth_rate << '\n';
					}
				}
				return 0;
			}

			vdf_diffusion<float>(
				params, Space<float>{vparall_axis, vperp_axis}, vparall_size, vperp_size,
				1.0e-10f, 0.f, //amplitude, amplitude time
				10000, 1.f,  //iterations, dt
				false,         //initial core dfc export
				true,         //dfc recalculation
				true, 1000,   //intermidiate growth rate export
				true, 1000    //intermidiate vdf export
			);
		}

		if (false) {
			auto params = init_parameters(0.95, 1.0 / 0.95, 1./6., -2.);
			unsigned vparall_size = 512; unsigned vperp_size = 3072;
			Axis<double> vparall_axis = construct_vparall_axis<double>(params,make_ZFunc<double>(1.e-5, 15.), vparall_size, -12., -0.96);
			Axis<double> vperp_axis = { -0.75e-2, 1.5e-2 };

			//analytical growth rate
			if (false) {
				auto zfunc = make_ZFunc<double>(1.e-5, 15.);
				auto moments = AnalyticalMoments<double>(params);
				auto vres_solver = ResonantVelocitySolver<double>(zfunc, params);
				auto dDdw = DispersionRelationOmegaDerivative<double>(zfunc, params);

				{
					ofstream ascii_os;
					ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
					ascii_os.exceptions(ios::badbit | ios::failbit);
					ascii_os.open("./data/growth-rate-analytical.txt");
					for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx) {
						auto vparall = vparall_axis(vparall_idx);
						auto wk_pair = vres_solver(vparall);
						auto k_betta = wk_pair.second * params.betta_root_c;
						auto growth_rate = -double(1.25331414) * (moments.GDerive(vparall) - moments.g(vparall) / k_betta) / dDdw(wk_pair.first, wk_pair.second);
						ascii_os << wk_pair.second << ' ' << vparall << ' ' << growth_rate << '\n';
					}
				}
				return 0;
			}

			vdf_diffusion<double>(
				params, Space<double>{vparall_axis, vperp_axis}, vparall_size, vperp_size,
				1.0e-5, 0., //amplitude, amplitude time
				4000, 1.,  //iterations, dt
				false,         //initial core dfc export
				false,         //dfc recalculation
				true, 1000,   //intermidiate growth rate export
				true, 1000    //intermidiate vdf export
			);
		}
	} 
	catch(std::exception const &ex) {
		cout << ex.what() << endl;
	}

	return 0;
}