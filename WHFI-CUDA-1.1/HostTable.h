#pragma once

#include <vector>

namespace iki { namespace table {
	template <typename T>
	struct HostTable {
		HostTable(unsigned row_count, unsigned row_size): row_count(row_count), row_size(row_size), host_data(row_count * row_size) { }

		HostTable(HostTable &&src) = default;
		HostTable &operator=(HostTable && src) = default;
		HostTable(HostTable const &src) = default;
		HostTable &operator=(HostTable const &src) = default;

		T operator()(unsigned row_idx, unsigned elm_idx) const {
			return host_data[row_idx + elm_idx * row_count];
		}

		T& operator()(unsigned row_idx, unsigned elm_idx) {
			return host_data[row_idx + elm_idx * row_count];
		}

		unsigned full_size() const {
			return row_count * row_size;
		}

		HostTable& swap_sizes() {
			std::swap(row_count, row_size);
			return *this;
		}

		T *data() {
			return host_data.data();
		}

		T const *data() const {
			return host_data.data();
		}

		unsigned row_count, row_size;
		std::vector<T> host_data;
	};
} /*table*/ } /*iki*/