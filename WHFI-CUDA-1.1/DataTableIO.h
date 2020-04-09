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
	auto vector_idx = iki::first(table.get_bounds());
	for (size_t scalar_idx = 0; scalar_idx != iki::size(table.get_bounds()); ++scalar_idx, iki::next(vector_idx,table.get_bounds())) {
		ascii_out << vector_idx << ' ' << table[scalar_idx] << '\n';
	}
	return ascii_out;
}

namespace iki {
	//binary
	template <unsigned Dim>
	std::ostream& write_binary(std::ostream &binary_os, iki::Bounds<Dim> const &bounds) {
		for (unsigned d = 0; d != Dim; ++d) {
			size_t bound = bounds[d];
			binary_os.write(reinterpret_cast<char const *>(&bound), sizeof(bound));
		}
		return binary_os;
	}

	template <typename T, unsigned Dim, unsigned Scale>
	std::ostream& write_binary(std::ostream &binary_os, iki::DataTable<T, Dim, Scale> const &table) {
		write_binary(binary_os, table.get_bounds())
			.write(reinterpret_cast<char const *>(table.raw_data()), sizeof(T) * size(table.get_bounds()) * Scale);
		return binary_os;
	}

	template <unsigned Dim>
	std::istream &read_binary(std::istream &binary_is, iki::Bounds<Dim> &bounds) {
		for (unsigned d = 0; d != Dim; ++d) {
			binary_is.read(reinterpret_cast<char*>(&bounds[d]), sizeof(bounds[d]));
		}
		return binary_is;
	}

	template <typename T, unsigned Dim, unsigned Scale>
	std::istream &read_binary(std::istream &binary_in, iki::DataTable<T, Dim, Scale> &table) {
		Bounds<Dim> bounds;
		read_binary(binary_in, bounds);
		table.set_bounds(bounds);
		binary_in.read(reinterpret_cast<char *>(table.raw_data()), sizeof(T) * size(table.get_bounds()) * Scale);
		return binary_in;
	}
}/*iki*/