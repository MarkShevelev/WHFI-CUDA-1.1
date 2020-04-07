#include <cuda_runtime.h>

 namespace iki { namespace math { namespace device { 
	 template <typename T>
	 __device__ void thomson_sweep(T *a, T *b, T *c, T *d, T *x, size_t size, size_t stride = 1) {
		 for (size_t idx = 1; idx != size; ++idx) {
			 size_t stride_idx = idx * stride;
			 T w = a[stride_idx] / b[stride_idx - stride];
			 b[stride_idx] = fma(-w, c[stride_idx - stride], b[stride_idx]);
			 d[stride_idx] = fma(-w, d[stride_idx - stride], d[stride_idx]);
		 }
		 x[stride * (size - 1)] = d[stride * (size - 1)] / b[stride * (size - 1)];

		 for (size_t idx = size - 2; idx != 0; --idx) {
			 size_t stride_idx = idx * stride;
			 x[stride_idx] = fma(-c[stride_idx], x[stride_idx + stride], d[stride_idx]) / b[stride_idx];
		 }
		 x[0] = fma(-c[0], x[stride], d[0]) / b[0];
	 }
 }/*device*/ }/*math*/ }/*iki*/