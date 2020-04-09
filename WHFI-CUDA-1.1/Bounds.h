#pragma once

#include "Index.h"

namespace iki { 
	template <unsigned Dim>
	struct Bounds final {
		Bounds() {
			std::fill(components.begin(), components.end(), 0u);
		}

		Bounds(std::initializer_list<size_t> l) {
			std::copy(l.begin(), l.end(), components.begin());
		}

		Bounds(Bounds<Dim> const &src) {
			std::copy(src.begin(), src.end(), components.begin());
		}

		Bounds(Bounds<Dim> &&src) {
			std::copy(src.begin(), src.end(), components.begin());
		}

		Bounds<Dim> &operator=(Bounds<Dim> const &src) {
			std::copy(src.begin(), src.end(), components.begin());
			return *this;
		}

		Bounds<Dim> &operator=(Bounds<Dim> &&src) {
			std::copy(src.begin(), src.end(), components.begin());
			return *this;
		}

		size_t operator[](unsigned d) const { return components[d]; }
		size_t &operator[](unsigned d) { return components[d]; }

		size_t size() const {
			size_t size = 1;
			for (unsigned d = 0; d != Dim; ++d)
				size *= components[d];
			return size;
		}

		Index<Dim> index_size() const {
			Index<Dim> vector_idx;
			vector_idx[0] = 1;
			for (unsigned d = 1; d != Dim; ++d)
				vector_idx[d] = vector_idx[d - 1] * components[d - 1];
			return vector_idx;
		}

		size_t scalar_index(Index<Dim> const &vector_index) const {
			return index_size() * vector_index;
		}

		Index<Dim> first() const {
			return Index<Dim>();
		}

		Index<Dim> last() const {
			Index<Dim> result;
			for (unsigned d = 0; d != Dim; ++d)
				result[d] = components[d] == 0 ? 0 : components[d] - 1;
			return result;
		}

		Index<Dim>& next(Index<Dim> &idx) const {
			bool carry_flag = true;
			for (unsigned d = 0u; d != Dim; ++d)
				if (carry_flag) {
					idx[d] += 1u;
					carry_flag = false;
					if (idx[d] == components[d]) {
						idx[d] = 0u;
						carry_flag = true;
					}
				}
			return idx;
		}

		Index<Dim> next(Index<Dim> const &idx) const {
			Index<Dim> tmp(idx);
			return next(tmp);
		}


	private:
		std::array<size_t, Dim> components;
	};

	template <unsigned Dim>
	bool operator==(Bounds<Dim> const &lha, Bounds<Dim> const &rha) {
		for (unsigned d = 0; d != Dim; ++d)
			if (lha[d] != rha[d]) return false;
		return true;
	}

	template <unsigned Dim>
	bool operator!=(Bounds<Dim> const &lha, Bounds<Dim> const &rha) {
		return !(lha == rha);
	}
}/*iki*/