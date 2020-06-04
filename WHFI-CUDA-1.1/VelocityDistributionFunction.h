#pragma once

#include "host_math_helper.h"
#include "PhysicalParameters.h"
#include "HostGrid.h"

#include <cmath>

namespace iki { namespace whfi { 
	template <typename T>
	struct muVDF {
	public:
		muVDF(PhysicalParameters<T> params) : p(params) { }

		T operator()(T vparall, T mu) const {
			T coeff_c = std::exp(-mu), coeff_h = std::exp(-mu * p.TcTh_ratio);
			return
				p.nc * coeff_c * std::exp(-T(0.5) * math::pow<2>(vparall - p.bulk_to_term_c))
				+ p.nh * math::pow<3>(std::sqrt(p.TcTh_ratio)) * coeff_h *
				std::exp(-T(0.5) * math::pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h));
		}

	private:
		PhysicalParameters<T> p;
	};

	//vperp fast
	template <typename T>
	grid::HostGrid<T> calculate_muVDF(PhysicalParameters<T> params, grid::Space<T> space, unsigned vparall_size, unsigned mu_size) {
		grid::HostGrid<T> vdf_grid(space, vparall_size, mu_size);
		muVDF<T> mu_vdf(params);
		for (unsigned row_idx = 0; row_idx != vparall_size; ++row_idx) {
			for (unsigned elm_idx = 0; elm_idx != mu_size; ++elm_idx) {
				auto argument = space(row_idx, elm_idx);
				vdf_grid.table(row_idx, elm_idx) = mu_vdf(argument[0], argument[1]);
			}
		}
		return vdf_grid;
	}
}/*whfi*/ }/*iki*/