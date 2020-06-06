#pragma once

#include "PhysicalParameters.h"
#include "ZFunc.h"

namespace iki { namespace whfi {
	template <typename T>
	struct DispersionRelationOmegaDerivative {
		T operator()(T omega, T k) {
			T arg_c = (omega - T(1.)) / (k * p.betta_root_c) - p.bulk_to_term_c;
			T arg_h = (omega - T(1.)) / (k * p.betta_root_h) - p.bulk_to_term_h;
			T Zc = Z(arg_c), Zh = Z(arg_h);
			return p.nc / (k * p.betta_root_c) * (-Zc + (omega / (k * p.betta_root_c) - p.bulk_to_term_c) * (Zc * arg_c + T(1.)))
				+ p.nh / (k * p.betta_root_h) * (-Zh + (omega / (k * p.betta_root_h) - p.bulk_to_term_h) * (Zh * arg_h + T(1.)));
		}

		DispersionRelationOmegaDerivative(ZFunc<T> Z, PhysicalParameters<T> p) : Z(Z), p(p) { }

	private:
		ZFunc<T> Z;
		PhysicalParameters<T> p;
	};

	template <typename T>
	struct DispersionRelationKDerivative {
		T operator()(T omega, T k) {
			T arg_c = (omega - T(1.)) / (k * p.betta_root_c) - p.bulk_to_term_c;
			T arg_h = (omega - T(1.)) / (k * p.betta_root_h) - p.bulk_to_term_h;
			T Zc = Z(arg_c), Zh = Z(arg_h);
			return T(2.) * k
				+ p.nc * (omega / (k * k * p.betta_root_c) + p.bulk_to_term_c) * Zc
				- p.nc * (omega / (k * p.betta_c) - p.bulk_to_term_c) * (Zc * arg_c + T(1.)) * (omega - T(1.)) / (k * k * p.betta_root_c)
				+ p.nh * (omega / (k * k * p.betta_root_h) + p.bulk_to_term_h) * Zh
				- p.nh * (omega / (k * p.betta_root_h) - p.bulk_to_term_h) * (Zh * arg_h + T(1.)) * (omega - T(1.)) / (k * k * p.betta_root_h);
		}

		DispersionRelationKDerivative(ZFunc<T> Z, PhysicalParameters<T> p) : Z(Z), p(p) { }

	private:
		ZFunc<T> Z;
		PhysicalParameters<T> p;
	};
}/*whfi*/ }/*iki*/