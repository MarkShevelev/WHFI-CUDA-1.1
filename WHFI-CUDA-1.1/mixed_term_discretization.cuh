#pragma once

#include "DeviceTable.cuh"

#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device {
	template <typename T>
	__device__ T along_mixed_term_discretization(table::device::DeviceTable<T> x_curr, table::device::DeviceTable<T> along_mixed_dfc, unsigned row_idx, unsigned elm_idx) {
		if (2 <= elm_idx && along_mixed_dfc.row_size - 3 >= elm_idx) {
			auto im2_grad = x_curr(row_idx + 1, elm_idx - 2) - x_curr(row_idx - 1, elm_idx - 2);
			auto im1_grad = x_curr(row_idx + 1, elm_idx - 1) - x_curr(row_idx - 1, elm_idx - 1);
			auto i_grad = x_curr(row_idx + 1, elm_idx) - x_curr(row_idx - 1, elm_idx);
			auto ip1_grad = x_curr(row_idx + 1, elm_idx + 1) - x_curr(row_idx - 1, elm_idx + 1);
			auto ip2_grad = x_curr(row_idx + 1, elm_idx + 2) - x_curr(row_idx - 1, elm_idx + 2);

			auto left_grad = along_mixed_dfc(row_idx, elm_idx - 1) * (9 * (i_grad + im1_grad) - (ip1_grad + im2_grad)) / 32;
			auto right_grad = along_mixed_dfc(row_idx, elm_idx) * (9 * (ip1_grad + i_grad) - (ip2_grad + im1_grad)) / 32;

			return right_grad - left_grad;
		}

		auto im1_grad = x_curr(row_idx + 1, elm_idx - 1) - x_curr(row_idx - 1, elm_idx - 1);
		auto i_grad = x_curr(row_idx + 1, elm_idx) - x_curr(row_idx - 1, elm_idx);
		auto ip1_grad = x_curr(row_idx + 1, elm_idx + 1) - x_curr(row_idx - 1, elm_idx + 1);

		auto left_grad = along_mixed_dfc(row_idx, elm_idx - 1) * (6 * im1_grad + 3 * i_grad - ip1_grad) / 16;
		auto right_grad = along_mixed_dfc(row_idx, elm_idx) * (6 * ip1_grad + 3 * i_grad - im1_grad) / 16;

		return right_grad - left_grad;
	}

	template <typename T>
	__device__ T perp_mixed_term_discretization(table::device::DeviceTable<T> x_curr, table::device::DeviceTable<T> perp_mixed_dfc, unsigned row_idx, unsigned elm_idx) {
		if (2 <= row_idx && perp_mixed_dfc.row_count - 3 >= row_idx) {
			auto jm2_grad = x_curr(row_idx - 2, elm_idx + 1) - x_curr(row_idx - 2, elm_idx - 1);
			auto jm1_grad = x_curr(row_idx - 1, elm_idx + 1) - x_curr(row_idx - 1, elm_idx - 1);
			auto j_grad = x_curr(row_idx, elm_idx + 1) - x_curr(row_idx, elm_idx - 1);
			auto jp1_grad = x_curr(row_idx + 1, elm_idx + 1) - x_curr(row_idx + 1, elm_idx - 1);
			auto jp2_grad = x_curr(row_idx + 2, elm_idx + 1) - x_curr(row_idx + 2, elm_idx - 1);

			auto left_q = perp_mixed_dfc(elm_idx, row_idx - 1) * (9 * (j_grad + jm1_grad) - (jp1_grad + jm2_grad)) / 32;
			auto right_q = perp_mixed_dfc(elm_idx, row_idx) * (9 * (jp1_grad + j_grad) - (jp2_grad + jm1_grad)) / 32;

			return right_q - left_q;
		}

		auto jm1_grad = x_curr(row_idx - 1, elm_idx + 1) - x_curr(row_idx - 1, elm_idx - 1);
		auto j_grad = x_curr(row_idx, elm_idx + 1) - x_curr(row_idx, elm_idx - 1);
		auto jp1_grad = x_curr(row_idx + 1, elm_idx + 1) - x_curr(row_idx + 1, elm_idx - 1);

		auto left_q = perp_mixed_dfc(elm_idx, row_idx - 1) * (6 * jm1_grad + 3 * j_grad - jp1_grad) / 16;
		auto right_q = perp_mixed_dfc(elm_idx, row_idx) * (6 * jp1_grad + 3 * j_grad - jm1_grad) / 16;

		return right_q - left_q;
	}
}/*device*/ }/*diffusion*/ }/*iki*/