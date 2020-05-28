#pragma once

#include "thomson_sweep_kernel.cuh"
#include "cycle_grid_transpose.cuh"
#include "forward_step_matrix_calculation_kernel.cuh"
#include "correction_forward_step_matrix_calculation_kernel.cuh"
#include "HostManagedDeviceTable.cuh"
#include "HostTable.h"
#include "HostDeviceTransfer.cuh"
#include "HostManagedDeviceTableTranspose.cuh"

#include <cuda_runtime.h>
#include <algorithm>

namespace iki {	namespace diffusion {
	template <unsigned TILE_SIZE, unsigned THREAD_COUNT, typename T>
	class TwoDimensionalMultithreadDiffusion final {
		using Table = table::test::HostManagedDeviceTable<T>;
		using HostTable = table::HostTable<T>;
	public:
		TwoDimensionalMultithreadDiffusion(
			HostTable const &init_host, HostTable const &perp_dfc_host, T perp_r, HostTable const &along_dfc_host, T along_r): 
			a(init_host.row_count, init_host.row_size), 
			b(init_host.row_count, init_host.row_size), 
			c(init_host.row_count, init_host.row_size), 
			d(init_host.row_count, init_host.row_size), 
			x_prev_transposed(init_host.row_size, init_host.row_count),
			x_next_transposed(init_host.row_size, init_host.row_count),
			x_prev(table::test::construct_from(init_host)), x_next(table::test::construct_from(init_host)),
			perp_dfc(table::test::construct_from(perp_dfc_host)), along_dfc(table::test::construct_from(along_dfc_host)), 
			perp_r(perp_r),	along_r(along_r) {
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("TwoDimensionalMultithreadDiffusion construction: ", cudaStatus);
		}

		TwoDimensionalMultithreadDiffusion& step() {
			{
				cudaError_t cudaStatus;
				unsigned row_count = x_prev.dTable.row_count, row_size = x_prev.dTable.row_size;

				dim3 blockDim(TILE_SIZE, TILE_SIZE), gridDim(row_count / TILE_SIZE, row_size / TILE_SIZE);
				device::forward_step_matrix_calculation_kernel<TILE_SIZE><<<gridDim, blockDim>>> (a.dTable, b.dTable, c.dTable, d.dTable, x_prev.dTable, along_dfc.dTable, along_r, perp_dfc.dTable, perp_r);
				cudaDeviceSynchronize();
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Forward step matrix calculation kernel: ", cudaStatus);

				math::device::thomson_sweep_kernel<<<row_count / THREAD_COUNT, THREAD_COUNT>>>(a.dTable, b.dTable, c.dTable, d.dTable, x_next.dTable);
				cudaDeviceSynchronize();
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Forward step Thomson sweep: ", cudaStatus);

				
			}

			table::test::transpose(x_prev, x_prev_transposed);
			table::test::transpose(x_next, x_next_transposed);
			a.swap_sizes(); b.swap_sizes(); c.swap_sizes(); d.swap_sizes();

			{
				cudaError_t cudaStatus;
				unsigned row_count = x_prev_transposed.dTable.row_count, row_size = x_prev_transposed.dTable.row_size;

				dim3 blockDim(TILE_SIZE, TILE_SIZE), gridDim(row_count / TILE_SIZE, row_size / TILE_SIZE);
				device::correction_forward_step_matrix_calculation_kernel<TILE_SIZE><<<gridDim, blockDim>>> (a.dTable, b.dTable, c.dTable, d.dTable, x_prev_transposed.dTable, x_next_transposed.dTable, perp_dfc.dTable, perp_r);
				cudaDeviceSynchronize();
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Correction step matrix calculation kernel: ", cudaStatus);

				math::device::thomson_sweep_kernel << <row_count / THREAD_COUNT, THREAD_COUNT >> > (a.dTable, b.dTable, c.dTable, d.dTable, x_next_transposed.dTable);
				cudaDeviceSynchronize();
				if (cudaSuccess != (cudaStatus = cudaGetLastError()))
					throw DeviceError("Correction step Thomson sweep kernel: ", cudaStatus);

				
			}

			table::test::transpose(x_prev_transposed, x_prev);
			table::test::transpose(x_next_transposed, x_next);
			a.swap_sizes(); b.swap_sizes(); c.swap_sizes(); d.swap_sizes();

			std::swap(x_prev, x_next);
			return *this;
		}

	public:
		Table a, b, c, d, x_prev_transposed, x_next_transposed;
		Table x_prev, x_next;
		Table perp_dfc, along_dfc;
		T const perp_r,along_r;
	};
} /*diffusion*/ } /*iki*/