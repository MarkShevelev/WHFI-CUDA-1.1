#pragma once

#include <stdexcept>

namespace iki { namespace math {
	template <typename T, typename Func_t>
	T step_solve(T min, T max, T step, Func_t f) { //[min,max)
		unsigned counter = 0u; 
		T arg_curr = min, arg_next = min + step,  arg_res = T(0.5) * (arg_curr + arg_next), 
			f_curr = f(arg_curr), f_next = f(arg_next);

		while (arg_res < max) {
			if (f_curr * f_next < 0.) { return arg_res; }
			arg_curr = arg_next; f_curr = f_next;
			arg_next = min + ++counter * step; f_next = f(arg_next);
			arg_res = T(0.5) * (arg_curr + arg_next);
		}
		throw std::runtime_error("Can't find root!");
	}
}/*math*/ } /*iki*/