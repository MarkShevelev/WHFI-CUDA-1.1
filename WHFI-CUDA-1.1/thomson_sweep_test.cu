#include "thomson_sweep.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>

template <typename T>
__device__ void initial_matrix(T *a, T *b, T *c, T *d, T *x, unsigned size) {
	for (unsigned offset = 0; offset != size; ++offset) {
		a[offset] = 1.f;
		b[offset] = 3.f;
		c[offset] = 1.f;
		d[offset] = 5.f;
		x[offset] = 0;
	}

	d[0] = d[size - 1] = 4.f;
}

template <typename T>
__global__ void thomson_sweep_single_thread_test_kernel(T *a, T *b, T *c, T *d, T *x, unsigned size) {
	initial_matrix(a, b, c, d, x, size);
	iki::math::device::thomson_sweep(a, b, c, d, x, size);
}

int main() {
	using namespace std;
	try {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaSetDevice(0)))
			throw runtime_error(cudaGetErrorString(cudaStatus));

		unsigned size = 1024;
		void *dev_memory = NULL;
		cudaStatus = cudaMalloc(&dev_memory, sizeof(float) * size * 5);
		if (cudaSuccess != cudaStatus)
			throw runtime_error(cudaGetErrorString(cudaStatus));

		float *a = (float*)dev_memory;
		float *b = a + size;
		float *c = b + size;
		float *d = c + size;
		float *x = d + size;

		dim3 threads_dim(1), block_dim(1);
		thomson_sweep_single_thread_test_kernel <<<threads_dim, block_dim>>> (a,b,c,d,x,size);

		vector<float> result(size);
		if (cudaSuccess != (cudaStatus = cudaMemcpy(result.data(), x, size * sizeof(float), cudaMemcpyDeviceToHost)))
			throw runtime_error(cudaGetErrorString(cudaStatus));

		{
			ofstream ascii_out("./data/result.txt");
			ascii_out.precision(7); ascii_out.setf(ios::fixed, ios::floatfield);
			for (auto const &x : result)
				ascii_out << x << '\n';
		}
			
	}
	catch (exception const &e) {
		cout << e.what() << endl;
	}


	return 0;
}