#pragma once

#include <array>
#include <numeric>

namespace iki {
	template <unsigned Dim>
	using Index = std::array<size_t, Dim>;

	template <unsigned Dim>
	size_t operator*(Index<Dim> const &lha, Index<Dim> const &rha) {
		return std::inner_product(lha.begin(), lha.end(), rha.begin(), 0u);
	}
} /*iki*/