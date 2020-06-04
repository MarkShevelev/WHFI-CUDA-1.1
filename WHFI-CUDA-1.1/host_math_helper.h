#pragma once

namespace iki { namespace math { 
	template <unsigned exp>
	float pow(float base) {
		return base * pow<exp - 1>(base);
	}

	template <>
	float pow<1>(float base) {
		return base;
	}

	template <unsigned exp>
	double pow(double base) {
		return base * pow<exp - 1>(base);
	}

	template <>
	double pow<1>(double base) {
		return base;
	}

}/*math*/ }/*iki*/