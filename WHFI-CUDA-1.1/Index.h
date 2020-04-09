#pragma once

#include <array>
#include <algorithm>
#include <initializer_list>

namespace iki {
	template <unsigned Dim>
	struct Index final {
		Index() {
			std::fill(componets.begin(), componets.end(), 0u);
		}

		Index(std::initializer_list<size_t> l) {
			std::copy(l.begin(), l.end(), componets.begin());
		}

		Index(Index<Dim> const &src) {
			std::copy(src.begin(), src.end(), componets.begin());
		}

		Index(Index<Dim> &&src) {
			std::copy(src.begin(), src.end(), componets.begin());
		}

		Index<Dim> &operator=(Index<Dim> const &src) {
			std::copy(src.begin(), src.end(), componets.begin());
			return *this;
		}

		Index<Dim> &operator=(Index<Dim> &&src) {
			std::copy(src.begin(), src.end(), componets.begin());
			return *this;
		}

		size_t operator[](unsigned d) const { return componets[d]; }
		size_t &operator[](unsigned d) { return componets[d]; }

	private:
		std::array<size_t, Dim> componets;
	};

	template <unsigned Dim>
	bool operator==(Index<Dim> const &lha, Index<Dim> const &rha) {
		for (unsigned d = 0; d != Dim; ++d)
			if (!(lha[d] == rha[d])) return false;
		return true;
	}

	template <unsigned Dim>
	bool operator!=(Index<Dim> const &lha, Index<Dim> const &rha) {
		return !(lha == rha);
	}

	template <unsigned Dim>
	size_t operator*(Index<Dim> const &lha, Index<Dim> const &rha) {
		size_t scalar_product = 1;
		for (unsigned d = 0; d != Dim; ++d)
			scalar_product += lha[d] * rha[d];
		return scalar_product;
	}
} /*iki*/