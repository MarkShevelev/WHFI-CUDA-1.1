#include "HostGrid.h"
#include "HostGridIO.h"
#include "VelocityDistributionFunction.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

int main() {
	

	iki::whfi::PhysicalParameters<float> params = iki::whfi::init_parameters(0.85f, 1.0f / 0.85f, 0.25f, -9.0f);
	iki::whfi::muVDF<float> mu_vdf(params);

	iki::grid::Space<float> vparall_mu_space = {
		iki::grid::Axis<float>{-16.5f, 1.5e-2},
		iki::grid::Axis<float>{0.75f, 1.5e-2}
	};
	auto vdf_grid = iki::whfi::calculate_muVDF(params, vparall_mu_space, 1024, 1024);

	try {
		{
			std::ofstream ascii_out;
			ascii_out.exceptions(std::ios::failbit | std::ios::badbit);

			ascii_out.open("./data/vdf-test.txt");
			ascii_out.precision(7);
			ascii_out.setf(std::ios::fixed, std::ios::floatfield);

			ascii_out << vdf_grid;
		}
	}
	catch (std::exception &ex) {
		std::cout << ex.what() << std::endl;
	}

	return 0;
}