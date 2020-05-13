#pragma once

#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device {
	template <typename T>
	inline
	__device__ T diagonal_discretization(T x_im1_j, T x_i_j, T x_ip1_j, T d_im1_j, T d_i_j) {
		return d_i_j * (x_ip1_j - x_i_j) - d_im1_j * (x_i_j - x_im1_j);
	}
}/*device*/ }/*diffusion*/ }/*iki*/