#pragma once

#include <vector>

namespace iki { namespace table {
	template <typename T>
	struct HostDataLine {
		HostDataLine(unsigned size): size(size), host_data(size) { }

		HostDataLine(HostDataLine &&src) = default;
		HostDataLine &operator=(HostDataLine && src) = default;
		HostDataLine(HostDataLine const &src) = default;
		HostDataLine &operator=(HostDataLine const &src) = default;

		T operator()(unsigned idx) const {
			return host_data[idx];
		}

		T& operator()(unsigned idx) {
			return host_data[idx];
		}

		T *data() {
			return host_data.data();
		}

		T const *data() const {
			return host_data.data();
		}

		unsigned size;
		std::vector<T> host_data;
	};
} /*table*/ } /*iki*/