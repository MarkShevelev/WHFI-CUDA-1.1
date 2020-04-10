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
std::ostream &operator<<(std::ostream &ascii_os, iki::table::DataTable<T, Dim, Scale> const &table) {
	auto vector_idx = iki::table::begin_index(table.get_bounds());
	for (size_t scalar_idx = 0; scalar_idx != iki::table::index_volume(table.get_bounds()); ++scalar_idx, iki::table::next_index(vector_idx,table.get_bounds())) {
		ascii_os << vector_idx << ' ' << table[scalar_idx] << '\n';
	}
	return ascii_os;
}

namespace iki { namespace table {
	template <typename T, unsigned Dim, unsigned Scale>
	std::ostream& write_binary(std::ostream &binary_os, DataTable<T, Dim, Scale> const &table) {
		write_binary(binary_os, table.get_bounds())
			.write(reinterpret_cast<char const *>(table.raw_data()), sizeof(T) * index_volume(table.get_bounds()) * Scale);
		return binary_os;
	}

	template <typename T, unsigned Dim, unsigned Scale>
	std::istream &read_binary(std::istream &binary_in, DataTable<T, Dim, Scale> &table) {
		Bounds<Dim> bounds;
		read_binary(binary_in, bounds);
		table.set_bounds(bounds);
		binary_in.read(reinterpret_cast<char *>(table.raw_data()), sizeof(T) * index_volume(table.get_bounds()) * Scale);
		return binary_in;
	}
}/*talbe*/ }/*iki*/