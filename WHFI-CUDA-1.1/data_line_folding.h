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
}/*whfi*/ }/*iki*/