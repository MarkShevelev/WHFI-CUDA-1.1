#include "PhysicalParameters.h"
#include "HostManagedDeviceTable.cuh"
#include "HostManagedDeviceDataLine.cuh"
#include "HostTable.h"
#include "HostDataLine.h"
#include "HostGrid.h"
#include "HostGridIO.h"
#include "ZFunc.h"
#include "ResonantVelocitySolver.h"
#include "DispersionRelationDerivative.h"
#include "host_math_helper.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace iki;
using namespace iki::whfi;
using namespace iki::table;
using namespace iki::grid;

ostream& configure(ostream &ascii_os) {
	ascii_os.exceptions(ios::failbit | ios::badbit);
	return ascii_os << setprecision(7) << scientific;
}

void float_diffusion_coefficient_test(PhysicalParameters<float> params, Axis<float> vparall_axis, Axis<float> vperp_axis, unsigned vparall_size, unsigned vperp_size) {
	auto R = HostDataLine<float>(vparall_size * 2);
	auto k_betta = HostDataLine<float>(vparall_size * 2);
	auto double_vparall_axis = Axis<float>{ vparall_axis.begin, vparall_axis.step * 0.5f };
	auto double_vperp_axis = Axis<float>{ vperp_axis.begin, vperp_axis.step * 0.5f };
	auto parall_shifted_axis = Axis<float>{ vparall_axis.begin + vparall_axis.step * 0.5f, vparall_axis.step };
	auto perp_shifted_axis = Axis<float>{ vperp_axis.begin + vperp_axis.step * 0.5f, vperp_axis.step };

	{
		auto zfunc = make_ZFunc(1.e-5f, 15.f);
		auto wk_solver = ResonantVelocitySolver<float>(zfunc, params);
		auto dr_w_derive = DispersionRelationOmegaDerivative<float>(zfunc, params);
		auto dr_k_derive = DispersionRelationKDerivative<float>(zfunc, params);

		for (unsigned idx = 0; idx != vparall_size * 2; ++idx) {
			auto v_res = double_vparall_axis(idx);
			auto wk_pair = wk_solver(v_res);
			auto dwdk = -dr_k_derive(wk_pair.first, wk_pair.second) / dr_w_derive(wk_pair.first, wk_pair.second);
			R(idx) = 1.0f / std::fabs(v_res - dwdk / params.betta_root_c) / params.betta_root_c;
			k_betta(idx) = wk_pair.second * params.betta_root_c;
		}
	}

	auto dfc_parall = HostGrid<float>(
		Space<float>{parall_shifted_axis, vperp_axis}, 
		vparall_size, vperp_size
	);
	auto dfc_perp = HostGrid<float>(
		Space<float>{vparall_axis, perp_shifted_axis}, 
		vparall_size, vperp_size
	);
	auto dfc_perp_parall = HostGrid<float>(
		Space<float>{parall_shifted_axis, vperp_axis},
		vparall_size, vperp_size
		);
	auto dfc_parall_perp = HostGrid<float>(
		Space<float>{vparall_axis, perp_shifted_axis},
		vparall_size, vperp_size
	);
	
	//vparall_shift vperp
	for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx)
		for (unsigned vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx) {
			dfc_parall.table(vparall_idx, vperp_idx) = vperp_axis(vperp_idx) * R(2 * vparall_idx + 1);
			dfc_perp_parall.table(vparall_idx, vperp_idx) = vperp_axis(vperp_idx) / k_betta(2 * vparall_idx + 1) * R(2 * vparall_idx + 1);
		}
	//vparall vperp_shift
	for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx)
		for (unsigned vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx) {
			dfc_perp.table(vparall_idx, vperp_idx) = perp_shifted_axis(vperp_idx) * R(2 * vparall_idx) / math::pow<2u>(k_betta(2 * vparall_idx));
			dfc_parall_perp.table(vparall_idx, vperp_idx) = perp_shifted_axis(vperp_idx) / k_betta(2 * vparall_idx) * R(2 * vparall_idx);
		}

	stringstream params_ss;
	params_ss.precision(2); params_ss.setf(ios::fixed, ios::floatfield);
	params_ss << params.betta_c << '-' << params.bulk_to_alfven_c;
	{
		{
			ofstream parall_parall_ascii_stream;
			parall_parall_ascii_stream << configure;
			parall_parall_ascii_stream.open("./data/dfc-parall-parall-" + params_ss.str() + "-test.txt");
			parall_parall_ascii_stream << dfc_parall;
		}

		{
			ofstream perp_perp_ascii_stream;
			perp_perp_ascii_stream << configure;
			perp_perp_ascii_stream.open("./data/dfc-perp-perp-" + params_ss.str() + "-test.txt");
			perp_perp_ascii_stream << dfc_perp;
		}

		{
			ofstream perp_parall_ascii_stream;
			perp_parall_ascii_stream << configure;
			perp_parall_ascii_stream.open("./data/dfc-perp-parall-" + params_ss.str() + "-test.txt");
			perp_parall_ascii_stream << dfc_perp_parall;
		}

		{
			ofstream parall_perp_ascii_stream;
			parall_perp_ascii_stream << configure;
			parall_perp_ascii_stream.open("./data/dfc-parall-perp-" + params_ss.str() + "-test.txt");
			parall_perp_ascii_stream << dfc_parall_perp;
		}
	}
}