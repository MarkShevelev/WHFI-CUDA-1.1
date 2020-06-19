#include "PhysicalParameters.h"
#include "HostGrid.h"
#include "construct_vparall_axis.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace iki;
using namespace iki::whfi;
using namespace iki::table;
using namespace iki::grid;

void float_automated_vparall_axis_test(PhysicalParameters<float> params, unsigned vparall_size, float begin, float end) {
	auto zfunc = make_ZFunc(1.e-5f, 15.f);
	auto vparall_axis = construct_vparall_axis<float>(params, zfunc, vparall_size, begin, end);
	
	auto moments = AnalyticalMoments<float>(params);
	auto vres_solver = ResonantVelocitySolver<float>(zfunc, params);
	auto dDdw = DispersionRelationOmegaDerivative<float>(zfunc, params);

	HostDataLine<float> growth_rate_line(vparall_size), omega_line(vparall_size), k_line(vparall_size);
	for (unsigned idx = 0; idx != vparall_size; ++idx) {
		auto vparall = vparall_axis(idx);
		auto wk_pair = vres_solver(vparall);
		auto k_betta = wk_pair.second * params.betta_root_c;
		growth_rate_line(idx) = -float(1.25331414) * (moments.GDerive(vparall) - moments.g(vparall) / k_betta) / dDdw(wk_pair.first, wk_pair.second);
		omega_line(idx) = wk_pair.first;
		k_line(idx) = wk_pair.second;
	}

	{
		ofstream ascii_os;
		ascii_os.exceptions(ios::failbit | ios::badbit);
		ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
		ascii_os.open("./data/automated-vparall-axis-test.txt");
		for (unsigned idx = 0; idx != vparall_size; ++idx) { 
			ascii_os << omega_line(idx) << ' ' << k_line(idx) << ' ' << growth_rate_line(idx) << '\n';
		}
	}
}