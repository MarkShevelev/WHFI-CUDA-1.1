#pragma once

#include "PhysicalParameters.h"
#include "HostGrid.h"
#include "ZFunc.h"
#include "AnalyticalMoments.h"
#include "ResonantVelocitySolver.h"
#include "DispersionRelationDerivative.h"


namespace iki { namespace whfi {
	template <typename T>
	grid::Axis<T> construct_vparall_axis(PhysicalParameters<T> params, ZFunc<T> zfunc, unsigned vparall_size, T begin, T end) {
		auto moments = AnalyticalMoments<T>(params);
		auto vres_solver = ResonantVelocitySolver<T>(zfunc, params);
		auto dDdw = DispersionRelationOmegaDerivative<T>(zfunc, params);

		auto growth_rate = [&moments, &vres_solver, &dDdw, &params] (T vparall) {
			auto wk_pair = vres_solver(vparall);
			auto k_betta = wk_pair.second * params.betta_root_c;
			return -T(1.25331414) * (moments.GDerive(vparall) - moments.g(vparall) / k_betta) / dDdw(wk_pair.first, wk_pair.second);
		};
		
		T left = begin, right = end, step = (right - begin) / (vparall_size - 1);
		grid::Axis<T> vparall_axis{ begin, step };
		while (growth_rate(vparall_axis(vparall_size-2)) < T(0.)) {
			auto center = (left + right) / T(2);
			auto growth_rate_center = growth_rate(center);
			while (growth_rate_center > 0) {
				left = center;
				center = (left + right) / T(2);
				growth_rate_center = growth_rate(center);
			}

			right = center;
			vparall_axis.step = (right - begin) / (vparall_size - 1);
		}
		return vparall_axis;
	}
}/*whfi*/ }/*iki*/