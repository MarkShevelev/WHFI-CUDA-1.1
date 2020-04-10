#pragma once

#include <array>
#include <algorithm>
#include <numeric>

namespace iki { namespace table {
	template <unsigned Dim>
	using Index = std::array<size_t, Dim>;

	template <unsigned Dim>
	size_t scalar_product(Index<Dim> const &lha, Index<Dim> const &rha) {
		return std::inner_product(lha.begin(), lha.end(), rha.begin(), (size_t)0);
	}

	template <unsigned Dim>
	using Bounds = std::array<size_t, Dim>;

	template <unsigned Dim>
	size_t index_volume(Bounds<Dim> const &bounds) {
		return std::accumulate(bounds.begin(), bounds.end(), (size_t)1, [] (auto lha, auto rha) { return lha * rha; });
	}

	template <unsigned Dim>
	Index<Dim> scalar_bounds(Bounds<Dim> const &bounds) {
		Index<Dim> vector_idx = { 1 };
		std::transform(bounds.begin(), std::prev(bounds.end()), std::next(vector_idx.begin()), [] (auto lha, auto rha) { return lha * rha; });
		return vector_idx;
	}

	template <unsigned Dim>
	Index<Dim> begin_index(Bounds<Dim> const &bounds) {
		Index<Dim> vector_idx = { 0 };
		return vector_idx;
	}

	template <unsigned Dim>
	Index<Dim> end_index(Bounds<Dim> const &bounds) {
		Index<Dim> vector_idx = { 0 };
		vector_idx[Dim - 1] = bounds[Dim - 1];
		return vector_idx;
	}

	template <unsigned Dim>
	size_t scalar_index(Index<Dim> const &vector_index, Bounds<Dim> const &bounds) {
		return scalar_product(vector_index, scalar_bounds(bounds));
	}

	template <unsigned Dim>
	Index<Dim> vector_index(size_t scalar_index, Bounds<Dim> const &bounds) {
		Index<Dim> result;
		std::transform(bounds.begin(), bounds.end(), result.begin(), [&scalar_index] (auto bound) { auto idx = scalar_index % bound; scalar_index /= bound; return idx; });
		return
	}

	template <unsigned Dim>
	Index<Dim>& next_index(Index<Dim> &idx, Bounds<Dim> const &bounds) {
		bool end = true;
		for (unsigned d = 0u; d != Dim; ++d)
			if (bounds[d] - 1 == idx[d])
				idx[d] = 0u;
			else {
				++idx[d];
				end = false;
				break;
			}
		return end ? end_index(bounds) : idx;
	}

	template <unsigned Dim>
	Index<Dim> next_index(Index<Dim> const &idx, Bounds<Dim> const &bounds) {
		Index<Dim> tmp_idx = idx;
		return next_index(tmp_idx);
	}

	template <unsigned Dim>
	Index<Dim>& prev_index(Index<Dim> &idx, Bounds<Dim> const &bounds) {
		bool end = true;
		for (unsigned d = 0u; d != Dim; ++d) {
			if (0u == idx[d])
				idx[d] = bounds[d] - 1;
			else {
				--idx[d];
				end = false;
				break;
			}
		}
		return end ? end_index(bounds) : idx;
	}

	template <unsigned Dim>
	Index<Dim> prev_index(Index<Dim> const &idx, Bounds<Dim> const &bounds) {
		Index<Dim> tmp_idx = idx;
		return prev_index(tmp_idx);
	}
}/*table*/ } /*iki*/