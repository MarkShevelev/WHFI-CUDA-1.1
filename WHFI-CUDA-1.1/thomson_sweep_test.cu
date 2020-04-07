#include "thomson_sweep.cuh"
#include "thomson_sweep_kernel.cuh"
#include "initial_matrix.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>



template <typename T>
__global__ void thomson_sweep_single_thread_test_kernel(T *a, T *b, T *c, T *d, T *x, unsigned size) {
	iki::math::device::thomson_sweep(a, b, c, d, x, size);
}

template <typename T>
__global__ void thomson_sweep_multithread_initial_matrix_kernel(T *a, T *b, T *c, T *d, T *x, unsigned row_size, unsigned row_count) {
	unsigned mtx_id = threadIdx.x + blockIdx.x * blockDim.x;
	initial_matrix(a + mtx_id, b + mtx_id, c + mtx_id, d + mtx_id, x + mtx_id, row_size, row_count, T(mtx_id + 1));
}

void thomson_sweep_single_thread_test(std::ostream &ascii_out) {
	using namespace std;

	cudaError_t cudaStatus;

	unsigned size = 1024;
	void *dev_memory = NULL;
	cudaStatus = cudaMalloc(&dev_memory, sizeof(float) * size * 5);
	if (cudaSuccess != cudaStatus)
		throw runtime_error(cudaGetErrorString(cudaStatus));

	float *a = (float *)dev_memory;
	float *b = a + size;
	float *c = b + size;
	float *d = c + size;
	float *x = d + size;

	dim3 threads_dim(1), block_dim(1);
	thomson_sweep_multithread_initial_matrix_kernel<<<block_dim, threads_dim>>>(a, b, c, d, x, size, 1);
	thomson_sweep_single_thread_test_kernel<<<threads_dim, block_dim>>>(a, b, c, d, x, size);
	if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
		stringstream error_stream;
		error_stream << "Can't launch 'thomson_sweep_single_thread_test_kernel'.\nCause:" << cudaGetErrorString(cudaStatus) << '\n';
		throw runtime_error(error_stream.str());
	}
	cudaDeviceSynchronize();

	vector<float> result(size);
	if (cudaSuccess != (cudaStatus = cudaMemcpy(result.data(), x, size * sizeof(float), cudaMemcpyDeviceToHost)))
		throw runtime_error(cudaGetErrorString(cudaStatus));
	
	for (auto const &x : result)
		ascii_out << x << '\n';

	cudaFree(dev_memory);
}

void thomson_sweep_multithread_test(std::ostream &ascii_out) {
	using namespace std;

	cudaError_t cudaStatus;

	unsigned const row_size = 1024, row_count = 2048, grid_size = row_size * row_count;
	void *dev_memory = NULL;
	cudaStatus = cudaMalloc(&dev_memory, grid_size * 5 * sizeof(float));
	if (cudaSuccess != cudaStatus)
		throw runtime_error(cudaGetErrorString(cudaStatus));

	float *a = (float *)dev_memory;
	float *b = a + grid_size;
	float *c = b + grid_size;
	float *d = c + grid_size;
	float *x = d + grid_size;

	dim3 block_dim(4), threads_dim(512);
	thomson_sweep_multithread_initial_matrix_kernel<<<block_dim, threads_dim>>>(a, b, c, d, x, row_size, row_count);
	iki::math::device::thomson_sweep_kernel<<<block_dim, threads_dim >>>(a, b, c, d, x, row_size, row_count);
	if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
		stringstream error_stream;
		error_stream << "Can't launch 'thomson_sweep_multithread_test_kernel'.\nCause:" << cudaGetErrorString(cudaStatus) << '\n';
		throw runtime_error(error_stream.str());
	}
	cudaDeviceSynchronize();

	{
		vector<float> result(grid_size);
		if (cudaSuccess != (cudaStatus = cudaMemcpy(result.data(), x, grid_size * sizeof(float), cudaMemcpyDeviceToHost)))
			throw runtime_error(cudaGetErrorString(cudaStatus));

		for (unsigned row_id = 0; row_id != row_count; ++row_id)
			for (unsigned x_id = 0; x_id != row_size; ++x_id)
				ascii_out << row_id << ' ' << x_id << ' ' << result[x_id * row_count + row_id] << '\n';
	}

	cudaFree(dev_memory);
}

int main() {
	using namespace std;

	try {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaSetDevice(0)))
			throw runtime_error(cudaGetErrorString(cudaStatus));
		{
			ofstream ascii_out;
			ascii_out.exceptions(ios::failbit | ios::badbit);
			ascii_out.open("./data/single_thread_test.txt");
			ascii_out.precision(7); ascii_out.setf(ios::fixed, ios::floatfield);

			thomson_sweep_single_thread_test(ascii_out);
		}

		{
			ofstream ascii_out;
			ascii_out.exceptions(ios::failbit | ios::badbit);
			ascii_out.open("./data/multithread_test.txt");
			ascii_out.precision(7); ascii_out.setf(ios::fixed, ios::floatfield);

			thomson_sweep_multithread_test(ascii_out);
		}
	}
	catch (exception const &e) {
		cout << e.what() << endl;
	}

	cudaDeviceReset();
	return 0;
}