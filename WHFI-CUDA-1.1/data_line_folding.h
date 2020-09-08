#pragma once

#include "HostDataLine.h"

namespace iki { namespace whfi {
	template <typename T>
	T four_points_on_uniform_grid(table::HostDataLine<T> const &line, T step) {
		T sum = T(0.); //3/8 formula
		for (unsigned idx = 1; idx + 3 < line.size; idx += 3) {
			sum += T(3. / 8.) * step * (line(idx) + T(3.) * line(idx + 1) + T(3.) * line(idx + 2) + line(idx + 3));
		}
		return sum;
	}

	template <typename T>
	T two_points_on_nonuniform_grid(table::HostDataLine<T> const &val, table::HostDataLine<T> const &arg) {
		T sum = T(0);
		for (unsigned idx = 1; idx < val.size; idx += 1) {
			sum += (val(idx - 1) + val(idx)) / 2 * (arg(idx) - arg(idx - 1));
		}
		return sum;
	}

	template <typename T>
	inline
	T lagrange_polynome_integral(T r1, T r2, T a, T b) {
		auto down = a * a * a / 3 - a * a / 2 * (r1 + r2) + r1 * r2 * a;
		auto up = b * b * b / 3 - b * b / 2 * (r1 + r2) + r1 * r2 * b;
		return up - down;
	}

	template <typename T>
	T three_points_on_nonumiform_grid(table::HostDataLine<T> const &val, table::HostDataLine<T> const &arg) {
		T sum = T(0);
		for (unsigned idx = 2; idx < val.size; idx += 2) {
			auto l1 = lagrange_polynome_integral(arg(idx - 2), arg(idx - 1), arg(idx - 2), arg(idx)) / (arg(idx) - arg(idx - 2)) / (arg(idx) - arg(idx - 1)) * val(idx);

			auto l2 = lagrange_polynome_integral(arg(idx - 1), arg(idx), arg(idx - 2), arg(idx)) / (arg(idx - 2) - arg(idx - 1)) / (arg(idx - 2) - arg(idx)) * val(idx - 2);

			auto l3 = lagrange_polynome_integral(arg(idx - 2), arg(idx), arg(idx - 2), arg(idx)) / (arg(idx - 1) - arg(idx - 2)) / (arg(idx - 1) - arg(idx)) * val(idx - 1);


			sum += l1 + l2 + l3;
		}
		return sum;
	}
}/*whfi*/ }/*iki*/