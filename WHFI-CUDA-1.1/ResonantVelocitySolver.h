#pragma once

#include "PhysicalParameters.h"
#include "ZFunc.h"
#include "step_solve.h"

#include <cmath>
#include <utility>
#include <optional>
#include <stdexcept>

namespace iki { namespace whfi {
	template <typename T>
	struct ResonantVelocitySolver {
		std::pair<T,T> operator()(T resonant_v) {
			auto Zc = Z(resonant_v - params.bulk_to_term_c)
				, Zh = Z(resonant_v * std::sqrt(params.TcTh_ratio) - params.bulk_to_term_h);
			auto dispersion_relation = [resonant_v, Zc, Zh,this] (T omega) {
				return T(1. / 1836.)
					+ (omega - 1) * (omega - 1) / (resonant_v * params.betta_root_c) / (resonant_v * params.betta_root_c)
					- params.nc * ((omega * resonant_v) / (omega - T(1.)) - params.bulk_to_term_c) * Zc
					- params.nh * ((omega * resonant_v * std::sqrt(params.TcTh_ratio)) / (omega - T(1.)) - params.bulk_to_term_h) * Zh;
			};

			auto omega = math::step_solve(T(0.), T(1.), T(1.e-5), dispersion_relation);
			return { omega, (omega - T(1.)) / (resonant_v * params.betta_root_c) };
		}

		ResonantVelocitySolver(ZFunc<T> Z, PhysicalParamenters<T> params): Z(Z), params(params) { }
	private:
		ZFunc<T> Z;
		PhysicalParamenters<T> params;

	};
} /* whfi */ } /* iki */