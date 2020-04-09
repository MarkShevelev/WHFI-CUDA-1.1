#pragma once

#include "Index.h"
#include "Bounds.h"

#include <vector>

namespace iki {
	template <typename Iterator>
	struct Range final {
		Range(Iterator begin, Iterator end): begin(begin), end(end) { }
		
		Iterator begin() const { begin; }
		Iterator end() const { end; }

		Iterator begin() { begin; }
		Iterator end() { end; }

	private:
		Iterator begin, end;
	};

	template <typename T, unsigned Dim, unsigned Scale>
	struct DataTable final {
		DataTable(Bounds<Dim> bounds): bounds(bounds), data(bounds.size()*Scale ) { }

		Range<std::vector<T>::const_iterator> operator[](Index<Dim> const &idx) const {
			auto offset = bounds.scalar_index(idx);
			return { data.cbegin() + offset*Scale, data.cbegin() + (offset + 1)*Scale };
		}

		Range<std::vector<T>::iterator> operator[](Index<Dim> const &idx) {
			auto offset = bounds.scalar_index(idx);
			return { data.begin() + offset, data.begin() + offset + Scale };
		}

		Range<std::vector<T>::const_iterator> operator[](size_t scalar_index) const {
			return { data.cbegin() + scalar_index * Scale, data.cbegin() + (scalar_index + 1) * Scale };
		}

		Range<std::vector<T>::iterator> operator[](size_t scalar_index) {
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