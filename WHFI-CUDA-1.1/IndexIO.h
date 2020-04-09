#pragma once

#include "Index.h"

#include <iostream>

template <unsigned Dim>
std::ostream &operator<<(std::ostream &ascii_out, iki::Index<Dim> const &vector_idx) {
	for (unsigned d = 0; d != Dim; ++d)
		ascii_out << vector_idx[d] << ' ';
	return ascii_out;
}