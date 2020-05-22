#pragma once

#include <vector>

namespace iki { namespace table {
	template <typename T>
	struct HostTable {
		HostTable(unsigned row_count, unsigned row_size): row_count(row_count), row_size(row_size), hData(row_count * row_size) { }

		HostTable(HostTable &&src) = default;
		HostTable &operator=(HostTable && src) = default;
		HostTable(HostTable const &src) = default;
		HostTable &operator=(HostTable const &src) = default;

		T operator()(unsigned row_idx, unsigned elm_idx) const {
			return hData[row_idx + elm_idx * row_count];
		}

		T& operator()(unsigned row_idx, unsigned elm_idx) {
			return hData[row_idx + elm_idx * row_count];
		}

		HostTable& swap_sizes() {
			std::swap(row_count, row_size);
			return *this;
		}

		unsigned row_count, row_size;
		std::vector<T> hData;
	};
} /*table*/ } /*iki*/