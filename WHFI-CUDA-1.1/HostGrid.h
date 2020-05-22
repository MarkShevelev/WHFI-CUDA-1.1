#pragma once

#include "HostTable.h"

#include <array>

namespace iki { namespace grid { namespace test {
	template <typename T>
	struct Axis {
		T operator()(unsigned idx) const { return begin + step * idx; }
		T begin, step;
	};

	template<typename T>
	using Argument = std::array<T,2>;

	template <typename T>
	struct Space {
		Argument<T> operator()(unsigned row_idx, unsigned elm_idx) const {
			return { perp(row_idx), along(elm_idx) };
		}
		Axis<T> perp, along;
	};

	template <typename T>
	struct HostGrid {
		HostGrid(Space<T> space, unsigned row_count, unsigned row_size) : table(row_count, row_size), space(space) { }

		table::HostTable<T> table;
		Space<T> space;
	};
}/*test*/ }/*grid*/ }/*iki*/
