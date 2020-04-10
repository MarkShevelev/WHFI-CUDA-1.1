#pragma once

#include "Index.h"

#include <iostream>

template <unsigned Dim>
std::ostream &operator<<(std::ostream &ascii_os, iki::table::Index<Dim> const &vector_idx) {
	for (auto x : vector_idx)
		ascii_os << x << ' ';
	return ascii_os;
}

namespace iki { namespace table {
	template <unsigned Dim>
	std::ostream &write_binary(std::ostream &binary_os, Bounds<Dim> const &bounds) {
		for (unsigned d = 0; d != Dim; ++d) {
			size_t bound = bounds[d];
			binary_os.write(reinterpret_cast<char const *>(&bound), sizeof(bound));
		}
		return binary_os;
	}

	template <unsigned Dim>
	std::istream &read_binary(std::istream &binary_is, Bounds<Dim> &bounds) {
		for (unsigned d = 0; d != Dim; ++d) {
			binary_is.read(reinterpret_cast<char *>(&bounds[d]), sizeof(bounds[d]));
		}
		return binary_is;
	}
}/*talbe*/ }/*iki*/