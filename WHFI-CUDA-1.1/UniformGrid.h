#pragma once

#include "DataTable.h"

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
		std::transform(vector_idx.begin(), vector_idx.end(), space.begin(), vector_argument.begin(), [] (auto idx, auto axis) { return axis.begin + axis.step * idx; });
		return vector_argument;
	}

	template <typename T, unsigned Dim, unsigned Scale>
	struct UniformGrid final {
		UniformGrid() { }
		UniformGrid(UniformSpace<T, Dim> space, table::Bounds<Dim> bounds): space(space), table(bounds) { }
		UniformSpace<T, Dim> space;
		table::DataTable<T, Dim, Scale> table;
	};

	template <typename T, unsigned Scale>
	UniformGrid<T, 2u, Scale> transposed_grid(UniformGrid<T, 2u, Scale> const &input_grid) {
		UniformGrid<T, 2u, Scale> output_grid;
		{
			output_grid.space[0] = input_grid.space[1];
			output_grid.space[1] = input_grid.space[0];

			table::Bounds<2u> output_bounds;
			output_bounds[0] = input_grid.table.get_bounds()[1];
			output_bounds[1] = input_grid.table.get_bounds()[0];
			output_grid.table.set_bounds(output_bounds);
		}

		for (unsigned row_id = 0; row_id != input_grid.table.get_bounds()[0]; ++row_id) {
			for (unsigned elm_id = 0; elm_id != input_grid.table.get_bounds()[1]; ++elm_id) {
				auto input_range = input_grid.table[row_id + elm_id * input_grid.table.get_bounds()[0]];
				auto output_range = output_grid.table[row_id * output_grid.table.get_bounds()[0] + elm_id];
				std::copy(input_range.begin(), input_range.end(), output_range.begin());
			}
		}
		return output_grid;
	}
}/*grid*/ }/*iki*/