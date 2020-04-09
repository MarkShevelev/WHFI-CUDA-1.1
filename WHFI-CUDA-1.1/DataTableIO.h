#pragma once

#include "DataTable.h"
#include "IndexIO.h"

#include <iostream>
#include <algorithm>

template <typename It>
std::ostream &operator<<(std::ostream &ascii_out, iki::Range<It> range) {
	for (auto &x : range)
		ascii_out << x << ' ';
	return ascii_out;
}

template <typename T, unsigned Dim, unsigned Scale>
std::ostream &operator<<(std::ostream &ascii_out, iki::DataTable<T, Dim, Scale> const &table) {
	auto vector_idx = table.get_bounds().first();
	for (size_t scalar_idx = 0; scalar_idx != table.get_bounds().size(); ++scalar_idx, table.get_bounds().next(vector_idx)) {
		ascii_out << vector_idx << ' ' << table[scalar_idx] << '\n';
	}
	return ascii_out;
}

namespace iki {
	//binary
	template <unsigned Dim>
	std::ostream& write_binary(std::ostream &binary_out, iki::Bounds<Dim> const &bounds) {
		for (unsigned d = 0; d != Dim; ++d) {
			size_t bound = bounds[d];
			binary_out.write(reinterpret_cast<char const *>(&bound), sizeof(bound));
		}
		return binary_out;
	}

	template <typename T, unsigned Dim, unsigned Scale>
	std::ostream& write_binary(std::ostream &binary_out, iki::DataTable<T, Dim, Scale> const &table) {
		write_binary(binary_out, table.get_bounds())
			.write(reinterpret_cast<char const *>(table.raw_data()), sizeof(T) * table.get_bounds().size() * Scale);
		return binary_out;
	}


}/*iki*/