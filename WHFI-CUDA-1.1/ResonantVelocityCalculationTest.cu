#include "ZFunc.h"
#include "ResonantVelocitySolver.h"
#include "HostGrid.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace iki;
using namespace iki::whfi;
using namespace iki::grid;
using namespace iki::table;

void float_resonant_velocity_calculation_test(PhysicalParameters<float> params, Axis<float> vparall_axis, Axis<float> vperp_axis, unsigned vparall_size, unsigned vperp_size) {
	auto wk_solver = iki::whfi::ResonantVelocitySolver<float>(iki::whfi::make_ZFunc(1.e-5f, 15.f), params);

	iki::grid::HostGridLine<float> w_line(vparall_axis, vparall_size);
	iki::grid::HostGridLine<float> k_line(vparall_axis, vparall_size);

	for (unsigned idx = 0; idx != vparall_size; ++idx) {
		auto wk_pair = wk_solver(vparall_axis(idx));
		w_line.line(idx) = wk_pair.first;
		k_line.line(idx) = wk_pair.second;
	}

	std::ofstream ascii_out;
	ascii_out.exceptions(std::ios::badbit | std::ios::failbit);

	{
		stringstream filename_ss;
		filename_ss.precision(2); filename_ss.setf(ios::fixed, ios::floatfield);
		filename_ss << "./data/vres-w-k-" << params.betta_c << '-' << params.bulk_to_alfven_c << "-test.txt";
		ascii_out.open(filename_ss.str());
	}
	
	ascii_out.precision(7); ascii_out.setf(std::ios::scientific, std::ios::floatfield);
	for (unsigned idx = 0; idx != vparall_size; ++idx) {
		ascii_out << vparall_axis(idx) << ' ' << w_line.line(idx) << ' ' << k_line.line(idx) << '\n';
	}
}