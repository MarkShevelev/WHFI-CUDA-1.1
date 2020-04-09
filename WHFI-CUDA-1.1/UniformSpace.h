#pragma once

#include <array>
#include <initializer_list>
#include <algorithm>

namespace iki {
	template <typename T>
	struct Axis final {
		T begin = { 0 }, step = { 0 };
	};

	template <typename T, unsigned Dim>
	struct UniformSpace final {
		UniformSpace() { }
		UniformSpace(std::initializer_list<Axis<T>> l) {
			std::copy(l.begin(), l.end(), axes.begin());
		}

		Axis<T> operator[](unsigned d) const {
			return axes[d];
		}

		Axis<T>& operator[](unsigned d) {
			return axes[d];
		}

	private:
		std::array<Axis<T>, Dim> axes;
	};
}/*iki*/