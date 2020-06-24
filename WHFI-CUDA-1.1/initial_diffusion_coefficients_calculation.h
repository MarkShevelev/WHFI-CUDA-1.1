#pragma once

#include "PhysicalParameters.h"
#include "HostTable.h"
#include "HostDataLine.h"
#include "HostGrid.h"
#include "HostGridIO.h"
#include "HostDataLine.h"
#include "host_math_helper.h"
#include "ResonantVelocitySolver.h"
#include "DispersionRelationDerivative.h"

template <typename T>
void initial_diffusion_coefficients_calculation(
	iki::whfi::PhysicalParameters<T> params, 
	iki::grid::Axis<T> vparall_axis, iki::grid::Axis<T> vperp_axis, 
	unsigned vparall_size, unsigned vperp_size, 
	iki::table::HostTable<T> &dfc_vperp_vperp, 
	iki::table::HostTable<T> &dfc_vparall_vparall, //transposed
	iki::table::HostTable<T> &dfc_vparall_vperp,
	iki::table::HostTable<T> &dfc_vperp_vparall,   //transposed
	iki::table::HostDataLine<T> &k_betta,
	iki::table::HostDataLine<T> &dispersion_derive
) {
	using namespace iki;
	using namespace iki::whfi;
	using namespace iki::table;
	using namespace iki::grid;

	auto R = HostDataLine<T>(vparall_size * 2);
	auto double_k_betta = HostDataLine<T>(vparall_size * 2);
	auto double_vparall_axis = Axis<T>{ vparall_axis.begin, vparall_axis.step * 0.5f };
	auto shift_vperp_axis = Axis<T>{ vperp_axis.begin + vperp_axis.step * 0.5f, vperp_axis.step };

	{
		auto zfunc = make_ZFunc<T>(T(1.e-5), T(15.));
		auto wk_solver = ResonantVelocitySolver<T>(zfunc, params);
		auto dr_w_derive = DispersionRelationOmegaDerivative<T>(zfunc, params);
		auto dr_k_derive = DispersionRelationKDerivative<T>(zfunc, params);

		for (unsigned idx = 0; idx != vparall_size * 2; ++idx) {
			auto v_res = double_vparall_axis(idx);
			auto wk_pair = wk_solver(v_res);
			auto dwdk = -dr_k_derive(wk_pair.first, wk_pair.second) / dr_w_derive(wk_pair.first, wk_pair.second);
			R(idx) = 1.0f / std::fabs(v_res - dwdk / params.betta_root_c) / params.betta_root_c;
			double_k_betta(idx) = wk_pair.second * params.betta_root_c;

			if (0 == idx % 2) {
				k_betta(idx / 2) = wk_pair.second * params.betta_root_c;
				dispersion_derive(idx / 2) = dr_w_derive(wk_pair.first, wk_pair.second);
			}
		}

		//vparall vperp_shift
		for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx) {
			for (unsigned vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx) {
				dfc_vperp_vperp(vparall_idx, vperp_idx) = shift_vperp_axis(vperp_idx) * R(2 * vparall_idx) / math::pow<2u>(double_k_betta(2 * vparall_idx));
				dfc_vparall_vperp(vparall_idx, vperp_idx) = vperp_axis(vperp_idx) / double_k_betta(2 * vparall_idx) * R(2 * vparall_idx);
			}
		}

		//vparall_shift vperp //transposed!!!
		for (unsigned vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx) {
			for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx) {
				dfc_vparall_vparall(vperp_idx, vparall_idx) = vperp_axis(vperp_idx) * R(2 * vparall_idx + 1);
				dfc_vperp_vparall(vperp_idx, vparall_idx) = vperp_axis(vperp_idx) / double_k_betta(2 * vparall_idx + 1) * R(2 * vparall_idx + 1);
			}
		}

		//boundary conditions: zero flow through vperp = 0
		for (unsigned vparall_idx = 0; vparall_idx != vparall_size; ++vparall_idx) {
			dfc_vperp_vperp(vparall_idx, 0) = dfc_vparall_vperp(vparall_idx, 0) = T(0);
			dfc_vparall_vparall(0, vparall_idx) = dfc_vperp_vparall(0, vparall_idx) = T(0);
		}

		//boundary conditions: zero flow through vparall = max
		for (unsigned vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx) {
			dfc_vparall_vparall(vperp_idx, vparall_size - 2) = dfc_vperp_vparall(vperp_idx, vparall_size - 1) = T(0);
			dfc_vperp_vperp(vparall_size - 1, vperp_idx) = dfc_vparall_vperp(vparall_size - 1, vperp_idx) = T(0);
		}
	}
}