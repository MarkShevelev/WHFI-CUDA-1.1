#pragma once

#include "DataTable.h"
#include "UniformSpace.h"

#include <algorithm>

namespace iki { namespace grid {
	template <typename T>
	struct Axis final {
		T begin = { 0 }, step = { 0 };
	};

	template <typename T, unsigned Dim>
	using UniformSpace = std::array<Axis<T>, Dim>;

	template <typename T, unsigned Dim>
	using Argument = std::array<T, Dim>;

	template <typename T, unsigned Dim>
	Argument<T,Dim> make_argument(table::Index<Dim> const &vector_idx, UniformSpace<T,Dim> const &space) {
		Argument<T, Dim> vector_argument;
		std::transform(vector_idx.begin(), vector_idx.end(), space.begin(), space.end(), vector_argument.begin(), [] (auto idx, auto axis) { return axis.begin + axis.step * idx; });
		return vector_argument;
	}

	template <typename T, unsigned Dim, unsigned Scale>
	struct UniformGrid final {
		UniformSpace<T, Dim> space;
		table::DataTable<T, Dim, Scale> table;
	};
}/*grid*/ }/*iki*/