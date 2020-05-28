#pragma once

#include "HostGrid.h"

#include <iostream>

template <typename T>
std::ostream &operator<<(std::ostream &ascii_os, iki::grid::Argument<T> const &arg) {
	return ascii_os << arg[0] << ' ' << arg[1];
}

template <typename T>
std::ostream &operator<<(std::ostream &ascii_os, iki::grid::HostGrid<T> const &grid) {
	for (unsigned row_idx = 0; row_idx != grid.table.row_count; ++row_idx)
		for (unsigned elm_idx = 0; elm_idx != grid.table.row_size; ++elm_idx)
			ascii_os << grid.space(row_idx, elm_idx) << ' ' << grid.table(row_idx, elm_idx) << '\n';
	return ascii_os;
}