#include <cuda_runtime.h>

template <typename T>
__device__ void initial_matrix(T *a, T *b, T *c, T *d, T *x, unsigned size, unsigned stride, T ans) {
	for (unsigned offset = 0; offset != size; ++offset) {
		a[offset * stride] = 1;
		b[offset * stride] = 3;
		c[offset * stride] = 1;
		d[offset * stride] = 5 * ans;
		x[offset * stride] = 0;
	}

	d[0] = d[stride * (size - 1)] = 4 * ans;
}