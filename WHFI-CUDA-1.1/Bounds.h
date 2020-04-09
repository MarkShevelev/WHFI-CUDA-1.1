#pragma once

#include "Index.h"

#include <iterator>
#include <algorithm>
#include <numeric>

namespace iki { 
	template <unsigned Dim>
	using Bounds = std::array<size_t, Dim>;

	template <unsigned Dim>
	size_t size(Bounds<Dim> const &bounds) {
		return std::accumulate(bounds.begin(), bounds.end(), 1u, [] (auto lha, auto rha) { return lha * rha; });
	}

	template <unsigned Dim>
	Index<Dim> index_size(Bounds<Dim> const &bounds) {
		Index<Dim> vector_idx = { 1 };
		std::transform(bounds.begin(), std::prev(bounds.end()), std::next(vector_idx.begin()), [] (auto lha, auto rha) { return lha * rha; } );
		return vector_idx;
	}

	template <unsigned Dim>
	size_t scalar_index(Index<Dim> const &vector_index, Bounds<Dim> const &bounds) {
		return vector_index * index_size(bounds);
	}

	template <unsigned Dim>
	Index<Dim> first(Bounds<Dim> const &bounds) {
		Index<Dim> vector_idx;
		std::fill(vector_idx.begin(), vector_idx.end(), 0u);
		return vector_idx;
	}

	template <unsigned Dim>
	Index<Dim> last(Bounds<Dim> const &bounds) {
		Index<Dim> vector_idx;
		std::fill(vector_idx.begin(), vector_idx.end(), 0u);
		vector_idx[Dim - 1] = bounds[Dim - 1];
		return vector_idx;
	}

	template <unsigned Dim>
	Index<Dim> &next(Index<Dim> &idx, Bounds<Dim> const &bounds) {
		bool carry_flag = true;
		for (unsigned d = 0u; d != Dim; ++d)
			if (carry_flag) {
				idx[d] += 1u;
				carry_flag = false;
				if (idx[d] == bounds[d]) {
					idx[d] = 0u;
					carry_flag = true;
				}
			}
		return carry_flag ? (idx = last(bounds)) : idx;
	}

	template <unsigned Dim>
	Index<Dim> next(Index<Dim> const &idx, Bounds<Dim> const &bounds) {
		Index<Dim> result_idx = idx;
		return next(result_idx);
	}
}/*iki*/