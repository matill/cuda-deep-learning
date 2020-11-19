#include "activation_func.h"


inline __device__ void apply_sigmoid(vector_t vector) {
    f32 *val = &vector.vals[threadIdx.x];
    *val = 1 / (1 + expf(-*val));
}


inline __device__ void apply_softmax(vector_t vector) {
    // TODO: Optimize
    f32 *val = &vector.vals[threadIdx.x];
    *val = expf(*val);
    __syncthreads();
    f32 sum = 0;
    for (u32 i = 0; i != vector.size; i++) {
        sum += vector.vals[i];
    }
    *val /= sum;
}


__device__ void apply_activation_func(vector_t vector, activation_func_t activation_func) {
    switch (activation_func) {
        case SOFTMAX:
            apply_softmax(vector);
            break;
        case SIGMOID:
            apply_sigmoid(vector);
            break;
    }
}


// y_gradient: vector that can be used to store all y_k_derivative values since softmax needs all values and not just the one
// corresponding to the same index. The buffer is uninitialized, since it is in device-global memory, and is therefore slower.
__device__ f32 transform_yk_derivative_to_vk_derivative(vector_t y, f32 y_k_derivative, vector_t y_gradient, activation_func_t activation_func) {
    u32 k = threadIdx.x;
    f32 y_k = y.vals[k];
    f32 sigma_c;
    switch (activation_func) {
        case SIGMOID:
            return y_k * (y_k - 1) * y_k_derivative;

        case SOFTMAX:
            // Temporarily store all y_k_derivative in shared memory to compute sigma_c.
            // TODO: Consider removing syncs if the vector_dot function is optimized
            y_gradient.vals[k] = y_k_derivative;
            __syncthreads();
            sigma_c = vector_dot(y_gradient, y);

            // Compute v derivatives according to formulas.
            // Sync to avoid modifying y_gradient buffer before all threads are done using it.
            __syncthreads();
            return y_k * (y_k_derivative  - sigma_c);

        default:
            printf("ERROR: activation_func: %d\n", activation_func);
            return 0;
    }
}

