#pragma once

#include "DataTable.h"
#include "UniformSpace.h"

namespace iki {
	template <typename T, unsigned Dim, unsigned Scale>
	struct UniformGrid final {


	private:
		UniformSpace<T, Dim> space;
		DataTable<T, Dim, Scale> table;
	};
}/*iki*/