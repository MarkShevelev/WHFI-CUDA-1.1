#pragma once

#include "Index.h"
#include "Bounds.h"

#include <iterator>
#include <vector>

namespace iki {
	template <typename It>
	struct Range {
		using value_type = typename std::iterator_traits<It>::value_type;

		Range(It begin, It end) : begin_(begin), end_(end) { }
		It begin() const { return begin_; }
		It end() const { return end_; }

	private:
		It begin_;
		It end_;
	};

	template <typename T, unsigned Dim, unsigned Scale>
	struct DataTable final {
		DataTable(Bounds<Dim> const &bounds): bounds(bounds), data(bounds.size()*Scale ) { }

		Bounds<Dim> const &get_bounds() const { return bounds; }

		Range<typename std::vector<T>::const_iterator> operator[](Index<Dim> const &idx) const {
			return (*this)[bounds.scalar_index(idx)];
		}

		Range<typename  std::vector<T>::iterator> operator[](Index<Dim> const &idx) {
			return (*this)[bounds.scalar_index(idx)];
		}

		Range<typename  std::vector<T>::const_iterator> operator[](size_t scalar_index) const {
			return { data.cbegin() + scalar_index * Scale, data.cbegin() + (scalar_index + 1) * Scale };
		}

		Range<typename  std::vector<T>::iterator> operator[](size_t scalar_index) {
			return { data.cbegin() + scalar_index * Scale, data.cbegin() + (scalar_index + 1) * Scale };
		}

		T *raw_data() {
			return data.data();
		}

	private:
		Bounds<Dim> const bounds;
		std::vector<T> data;
	};
}