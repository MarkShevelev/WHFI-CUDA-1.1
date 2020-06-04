#include "ZFunc.h"
#include "ResonantVelocitySolver.h"
#include "HostGrid.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

int float_fesonant_velocity_calculation_test() {
	iki::whfi::PhysicalParameters<float> params = iki::whfi::init_parameters(0.85f, 1.0f / 0.85f, 0.25f, -9.0f);
	auto wk_solver = iki::whfi::ResonantVelocitySolver<float>(iki::whfi::make_ZFunc(1.e-4f, 10.f), params);

	unsigned vparall_size = 1024;
	iki::grid::Axis<float> vparall_axis = { -16.5f, 1.5e-2 };
	iki::grid::HostGridLine<float> w_line(vparall_axis, vparall_size);
	iki::grid::HostGridLine<float> k_line(vparall_axis, vparall_size);

	try {
		auto solver = iki::whfi::ResonantVelocitySolver<float>(iki::whfi::make_ZFunc(1.e-5f, 15.f), params);
		for (unsigned idx = 0; idx != vparall_size; ++idx) {
			auto wk_pair = solver(vparall_axis(idx));
			w_line.line(idx) = wk_pair.first;
			k_line.line(idx) = wk_pair.second;
		}

		std::ofstream ascii_out;
		ascii_out.exceptions(std::ios::badbit | std::ios::failbit);

		ascii_out.open("./data/vres-omega-k-test.txt");
		ascii_out.precision(7); ascii_out.setf(std::ios::fixed, std::ios::floatfield);
		for (unsigned idx = 0; idx != vparall_size; ++idx) {
			ascii_out << vparall_axis(idx) << ' ' << w_line.line(idx) << ' ' << k_line.line(idx) << '\n';
		}
	}
	catch (std::exception const &ex) {
		std::cout << ex.what() << std::endl;
	}

	return 0;
}