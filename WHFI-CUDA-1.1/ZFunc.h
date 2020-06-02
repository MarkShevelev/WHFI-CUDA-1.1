#pragma once

#include <vector>
	
namespace iki {	namespace whfi {
	template <typename T, typename F_t>
	struct Runge4th {
		T operator()(size_t x_cnt, T y) const {
			T x = x0 + x_cnt * dx;
			T k1 = F(x, y);
			T k2 = F(x + dx / T(2.), y + k1 * dx / T(2.));
			T k3 = F(x + dx / T(2.), y + k2 * dx / T(2.));
			T k4 = F(x + dx, y + k2 * dx);
			return dx / T(6.) * (k1 + T(2.) * k2 + T(2.) * k3 + k4);
		}

		Runge4th(T x0, T dx, F_t F) : x0(x0), dx(dx), F(F) { }

	private:
		T const x0, dx;
		F_t F;
	};

	template <typename T>
	struct ZFuncODE {
		T operator()(T x, T y) const { return -(x * y + T(1.)); }
	};

	template <typename T>
	std::vector<T> ZFuncTableCalculator(T step, T max) {
		std::vector<T> table(static_cast<unsigned>(max / step) + 1);
		auto runge4th = Runge4th<T, ZFuncODE<T>>(T(0.), T(step), ZFuncODE<T>());
		T c = T(0), s = T(0);
		for (unsigned idx = 0; idx != table.size(); ++idx) {
			table[idx] = s;
			T y, t;
			y = runge4th(idx, s) - c;
			t = s + y;
			c = (t - s) - y;
			s = t;
		}
		return table;
	}

	template <typename T>
	struct ZFunc {
		ZFunc(T step, std::vector<T> table): step(step), table(std::move(table)) { }
		T operator()(T arg) const {
			T farg = std::fabs(arg);
			auto idx = static_cast<unsigned>(farg / step);
			if (0 == idx) {
				T m = farg * table[1] / step;
			}
			else if ((idx + 1) < table.size() ) {
				T a = 0.5 * (table[idx - 1] - 2 * table[idx] + table[idx + 1]) / (step * step);
				T b = 0.5 * (table[idx + 1] - table[idx - 1]) / step;
				T m = a * (farg - step * idx) * (farg - step * idx) + b * (farg - step * idx) + table[idx];
				return arg > T(0) ? m : -m;
			}
			else { //asymptotic
				T over = T(1.) / arg, square = over * over;
				return -over * (T(1.) + square + T(3.) * square * square);
			}
		}
	
		T const step;
		std::vector<T> table;
	};

	template <typename T>
	ZFunc<T> make_ZFunc(T step, T max) {
		return ZFunc<T>(step, ZFuncTableCalculator(step,max));
	}
} /* whfi */ } /* iki */