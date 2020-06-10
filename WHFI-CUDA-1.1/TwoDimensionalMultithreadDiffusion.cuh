#pragma once

#include "thomson_sweep_kernel.cuh"
#include "forward_step_matrix_calculation_kernel.cuh"
#include "correction_step_matrix_calculation_kernel.cuh"
#include "compensation_matrix_calculation_kernel.cuh"
#include "HostManagedDeviceTable.cuh"
#include "HostTable.h"
#include "HostDeviceTransfer.cuh"
#include "HostManagedDeviceTableTranspose.cuh"

#include <cuda_runtime.h>
#include <algorithm>

namespace iki {	namespace diffusion {
	template <unsigned TILE_SIZE, unsigned THREAD_COUNT, typename T>
	class TwoDimensionalMultithreadDiffusion final {
		using Table = table::HostManagedDeviceTable<T>;
		using HostTable = table::HostTable<T>;
	public:
		TwoDimensionalMultithreadDiffusion(
			HostTable const &init_host, 
			HostTable const &perp_dfc_host, T perp_r, 
			HostTable const &along_dfc_host, T along_r, 
			HostTable const &perp_mixed_dfc_host,  //along perp
			HostTable const &along_mixed_dfc_host, //perp along
			T mixed_r
		):
			a(init_host.row_count, init_host.row_size), 
			b(init_host.row_count, init_host.row_size), 
			c(init_host.row_count, init_host.row_size), 
			d(init_host.row_count, init_host.row_size), 
			x_prev_transposed(init_host.row_size, init_host.row_count),
			x_next_transposed(init_host.row_size, init_host.row_count),
			x_prev(table::construct_from(init_host)), x_next(table::construct_from(init_host)),
			perp_dfc(table::construct_from(perp_dfc_host)), along_dfc(table::construct_from(along_dfc_host)), perp_mixed_dfc(table::construct_from(perp_mixed_dfc_host)), along_mixed_dfc(table::construct_from(along_mixed_dfc_host)), 
			perp_r(perp_r), along_r(along_r), mixed_r(mixed_r) {
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("TwoDimensionalMultithreadDiffusion construction: ", cudaStatus);
		}

		//x_prev x_next
		void forward_step() {
			cudaError_t cudaStatus;
			dim3 blockDim(TILE_SIZE, TILE_SIZE), gridDim(x_prev.row_count / TILE_SIZE, x_prev.row_size / TILE_SIZE);
			device::forward_step_matrix_calculation_kernel<TILE_SIZE><<<gridDim, blockDim>>>(
				a.table(), b.table(), c.table(), d.table(), 
				x_prev.table(), 
				along_dfc.table(), along_r, 
				perp_dfc.table(), perp_r,
				along_mixed_dfc.table(), perp_mixed_dfc.table(), mixed_r
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("Forward step matrix calculation kernel: ", cudaStatus);

			math::device::thomson_sweep_kernel<<<x_prev.row_count / THREAD_COUNT, THREAD_COUNT>>>(
				a.table(), b.table(), c.table(), d.table(),
				x_next.table()
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("Forward step Thomson sweep: ", cudaStatus);
		}

		//x_prev_transposed x_next_transposed
		void correction_step() {
			cudaError_t cudaStatus;
			dim3 blockDim(TILE_SIZE, TILE_SIZE), gridDim(x_prev_transposed.row_count / TILE_SIZE, x_prev_transposed.row_size / TILE_SIZE);
			device::correction_step_matrix_calculation_kernel<TILE_SIZE><<<gridDim, blockDim>>>(
				a.table(), b.table(), c.table(), d.table(),
				x_prev_transposed.table(), x_next_transposed.table(),
				perp_dfc.table(), perp_r
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("Correction step matrix calculation kernel: ", cudaStatus);

			math::device::thomson_sweep_kernel<<<x_prev_transposed.row_count / THREAD_COUNT, THREAD_COUNT>>>(
				a.table(), b.table(), c.table(), d.table(),
				x_next_transposed.table()
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("Correction step Thomson sweep kernel: ", cudaStatus);
		}

		//x_prev x_next
		void compensated_step() {
			cudaError_t cudaStatus;
			dim3 blockDim(TILE_SIZE, TILE_SIZE), gridDim(x_prev.row_count / TILE_SIZE, x_prev.row_size / TILE_SIZE);

			device::forward_step_matrix_calculation_kernel<TILE_SIZE><<<gridDim, blockDim>>>(
				a.table(), b.table(), c.table(), d.table(),
				x_prev.table(),
				along_dfc.table(), along_r,
				perp_dfc.table(), perp_r,
				along_mixed_dfc.table(), perp_mixed_dfc.table(), mixed_r
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("Compensated step forward matrix calculation kernel: ", cudaStatus);

			device::compensation_matrix_calculation_kernel<TILE_SIZE><<<gridDim, blockDim>>>(
				d.table(),
				x_prev.table(), x_next.table(),
				along_mixed_dfc.table(), perp_mixed_dfc.table(), mixed_r
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("Compensation matrix calculation kernel: ", cudaStatus);

			math::device::thomson_sweep_kernel<<<x_prev.row_count / THREAD_COUNT, THREAD_COUNT>>>(
				a.table(), b.table(), c.table(), d.table(),
				x_next.table()
			);
			cudaDeviceSynchronize();
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw DeviceError("Compensated forward step Thomson sweep: ", cudaStatus);
		}

		//from x_prev to x_prev_transposed
		void forward_transpose() {
			table::transpose(x_prev, x_prev_transposed);
			table::transpose(x_next, x_next_transposed);
			a.swap_sizes(); b.swap_sizes(); c.swap_sizes(); d.swap_sizes();
		}

		//from x_prev_transposed to x_prev
		void back_transpose() {
			table::transpose(x_prev_transposed, x_prev);
			table::transpose(x_next_transposed, x_next);
			a.swap_sizes(); b.swap_sizes(); c.swap_sizes(); d.swap_sizes();
		}

		TwoDimensionalMultithreadDiffusion& step() {
			forward_step();
			forward_transpose();

			correction_step();
			back_transpose();

			compensated_step();
			forward_transpose();

			correction_step();
			back_transpose();

			std::swap(x_prev, x_next);
			return *this;
		}

	public:
		Table a, b, c, d, x_prev_transposed, x_next_transposed;
		Table x_prev, x_next;
		Table perp_dfc, along_dfc, perp_mixed_dfc, along_mixed_dfc; //perp_* should be pretransposed!
		T const perp_r, along_r, mixed_r;
	};
} /*diffusion*/ } /*iki*/