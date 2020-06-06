#pragma once

namespace iki { namespace math { 
	template <unsigned exp>
	inline
	float pow(float base) {
		return base * pow<exp - 1>(base);
	}

	template <>
	inline
	float pow<1>(float base) {
		return base;
	}

	template <unsigned exp>
	inline
	double pow(double base) {
		return base * pow<exp - 1>(base);
	}

	template <>
	inline
	double pow<1>(double base) {
		return base;
	}

}/*math*/ }/*iki*/