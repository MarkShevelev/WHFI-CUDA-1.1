#pragma once

#include "PhysicalParameters.h"
#include "HostGrid.h"
#include "host_math_helper.h"

#include <vector>
#include <cmath>

namespace iki { namespace whfi { 
	template <typename T>
	class AnalyticalMoments final {
	public:
		AnalyticalMoments(PhysicalParameters<T> p) : p(p) { }

		T g(T vparall) const {
			return p.nc * std::exp(-math::pow<2>(vparall - p.bulk_to_term_c) / T(2)) + p.nh * std::sqrt(p.TcTh_ratio) * exp(-math::pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h) / T(2));
		}

		T G(T vparall) const {
			return p.nc * std::exp(-math::pow<2>(vparall - p.bulk_to_term_c) / T(2)) + p.nh * std::sqrt(T(1) / p.TcTh_ratio) * exp(-math::pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h) / T(2));
		}

		T GDerive(T vparall) const {
			return -(vparall - p.bulk_to_alfven_c) * p.nc * std::exp(-math::pow<2>(vparall - p.bulk_to_term_c) / T(2)) - (vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h) * p.nh * exp(-math::pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h) / T(2));
		}

	private:
		iki::whfi::PhysicalParameters<T> p;
	};

	template <typename T>
	class AnalyticalMomentsTable final {
	public:
		AnalyticalMomentsTable(PhysicalParameters<T> p) : moments(p) { }

		std::vector<T> g(grid::Axis<T> vparall_axis, unsigned size) const {
			std::vector<T> table(size);
			for (unsigned idx = 0u; idx < size; ++idx)
				table[idx] = moments.g(vparall_axis(idx));
			return table;
		}

		std::vector<T> G(grid::Axis<T> vparall_axis, unsigned size) const {
			std::vector<T> table(size);
			for (unsigned idx = 0u; idx < size; ++idx)
				table[idx] = moments.G(vparall_axis(idx));
			return table;
		}

		std::vector<T> GDerive(grid::Axis<T> vparall_axis, unsigned size) const {
			std::vector<T> table(size);
			for (unsigned idx = 0u; idx < size; ++idx)
				table[idx] = moments.GDerive(vparall_axis(idx));
			return table;
		}

	private:
		AnalyticalMoments<T> moments;
	};
} /* whfi */ } /* iki */