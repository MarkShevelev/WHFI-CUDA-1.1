#pragma once

#include "UniformGrid.h"

#include <iostream>

template <typename T, unsigned Dim>
std::ostream &operator<<(std::ostream &ascii_os, iki::grid::Argument<T, Dim> const &argument) {
	for (auto arg : argument)
		ascii_os << arg << ' ';
	return ascii_os;
}

template <typename T, unsigned Dim, unsigned Scale>
std::ostream &operator<<(std::ostream &ascii_os, iki::grid::UniformGrid<T, Dim, Scale> const &grid) {
	auto vector_idx = iki::table::begin_index(grid.table.get_bounds());
	for (size_t scalar_idx = 0; scalar_idx != iki::table::index_volume(grid.table.get_bounds()); ++scalar_idx, iki::table::next_index(vector_idx, grid.table.get_bounds())) {
		ascii_os << iki::grid::make_argument(vector_idx,grid.space) << ' ' << grid.table[scalar_idx] << '\n';
	}
	return ascii_os;
}

namespace iki { namespace grid {
	template <typename T, unsigned Dim>
	std::ostream &write_binary(std::ostream &binary_os, iki::grid::UniformSpace<T, Dim> const &space) {
		for (auto axis : space) {
			binary_os.write(reinterpret_cast<char const *>(&axis.begin), sizeof(T));
			binary_os.write(reinterpret_cast<char const *>(&axis.step), sizeof(T));
		}
		return binary_os;
	}

	template <typename T, unsigned Dim, unsigned Scale>
	std::ostream &write_binary(std::ostream &binary_os, iki::grid::UniformGrid<T, Dim, Scale> const &grid) {
		iki::grid::write_binary(binary_os, grid.space);
		iki::table::write_binary(binary_os, grid.table);
		return binary_os;
	}

	template <typename T, unsigned Dim>
	std::istream &read_binary(std::istream &binary_is, iki::grid::UniformSpace<T, Dim> &space) {
		for (auto &axis : space) {
			binary_is.read(reinterpret_cast<char*>(&axis.begin), sizeof(T));
			binary_is.read(reinterpret_cast<char*>(&axis.step), sizeof(T));
		}
		return binary_is;
	}

	template <typename T, unsigned Dim, unsigned Scale>
	std::istream &read_binary(std::istream &binary_is, iki::grid::UniformGrid<T, Dim, Scale> &grid) {
		iki::grid::read_binary(binary_is, grid.space);
		iki::table::read_binary(binary_is, grid.table);
		return binary_is;
	}
}/*grid*/ }/*iki*/