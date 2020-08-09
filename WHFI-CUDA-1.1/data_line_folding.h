#pragma once

#include "HostDataLine.h"

namespace iki { namespace whfi {
	template <typename T>
	T data_line_folding(table::HostDataLine<T> const &line, T step) {
		T sum = T(0.); //3/8 formula
		for (unsigned idx = 1; idx + 3 < line.size; idx += 3) {
			sum += T(3. / 8.) * step * (line(idx) + T(3.) * line(idx + 1) + T(3.) * line(idx + 2) + line(idx + 3));
		}
		return sum;
	}
}/*whfi*/ }/*iki*/