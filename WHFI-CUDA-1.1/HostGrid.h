#pragma once

#include "HostTable.h"
#include "HostDataLine.h"

#include <array>
#include <algorithm>

namespace iki { namespace grid {
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

		Space &swap_axes() {
			std::swap(perp, along);
			return *this;
		}

		Axis<T> perp, along;
	};

	template <typename T>
	struct HostGrid {
		HostGrid(Space<T> space, table::HostTable<T> &&table): table(table), space(space) { }
		HostGrid(Space<T> space, unsigned row_count, unsigned row_size) : table(row_count, row_size), space(space) { }

		table::HostTable<T> table;
		Space<T> space;
	};

	template <typename T>
	struct HostGridLine {
		HostGridLine(Axis<T> axis, table::HostDataLine<T> &&line): line(line), axis(axis) { }
		HostGridLine(Axis<T> axis, unsigned size) : line(size), axis(axis) { }

		table::HostDataLine<T> line;
		Axis<T> axis;
	};
}/*grid*/ }/*iki*/
