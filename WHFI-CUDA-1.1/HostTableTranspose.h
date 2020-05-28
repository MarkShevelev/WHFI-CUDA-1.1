#pragma once

#include "HostTable.h"

namespace iki { namespace table { 
	template <typename T>
	void transpose(HostTable<T> const &from, HostTable<T> &to) {
		to.swap_sizes();
		for (unsigned row_idx = 0; row_idx != from.row_count; ++row_idx)
			for (unsigned elm_idx = 0; elm_idx != from.row_size; ++elm_idx)
				to(elm_idx, row_idx) = from(row_idx, elm_idx);
	}
}/*table*/ }/*iki*/