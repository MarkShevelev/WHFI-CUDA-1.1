#include <cuda_runtime.h>

template <typename T>
__device__ void initial_matrix(T *a, T *b, T *c, T *d, T *x, unsigned size, unsigned stride, T ans) {
	for (unsigned offset = 0; offset != size; ++offset) {
		a[offset] = 1;
		b[offset] = 3;
		c[offset] = 1;
		d[offset] = 5 * ans;
		x[offset] = 0;
	}

	d[0] = d[size - 1] = 4 * ans;
}