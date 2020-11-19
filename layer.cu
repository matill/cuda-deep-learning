#include "layer.h"
#include "common.h"
#include "linalg.h"
#include "cost_func.h"


void layer_init(layer_t *layer, u32 in_dimension, u32 out_dimension, activation_func_t activation_func) {
    layer->in_dimension = in_dimension;
    layer->out_dimension = out_dimension;
    layer->activation_func = activation_func;
    matrix_init(&layer->weights, out_dimension, in_dimension);
    vector_init(&layer->bias, out_dimension);
}


__device__ void apply_sigmoid(vector_t vector) {
    f32 *val = &vector.vals[threadIdx.x];
    *val = 1 / (1 + expf(-*val));
}


__device__ void apply_softmax(vector_t vector) {
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


__global__ void layer_compute(layer_t layer, vector_t in_vector, vector_t out_vector) {
    ASSERT_EQ_INT(layer.in_dimension, in_vector.size);
    ASSERT_EQ_INT(layer.out_dimension, out_vector.size);
    matrix_t weights = layer.weights;
    vector_t bias = layer.bias;

    // out_vector = weights * in_vector + bias
    matrix_vector_multiply(weights, in_vector, out_vector);
    out_vector.vals[threadIdx.x] += bias.vals[threadIdx.x];

    switch (layer.activation_func) {
        case SOFTMAX:
            apply_softmax(out_vector);
            break;
        case SIGMOID:
            apply_sigmoid(out_vector);
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


__global__ void compute_v_gradient_from_v_plus_gradient(vector_t v_gradient, vector_t v_plus_gradient, matrix_t w_plus, vector_t y, activation_func_t activation_func) {
    // Compute dJ / dy(r)
    u32 k = threadIdx.x;
    f32 y_k_derivarive = 0;
    u32 num_iters = w_plus.height;
    for (u32 i = 0; i != num_iters; i++) {
        y_k_derivarive += v_plus_gradient.vals[i] * *matrix_index(w_plus, i, k);
    }

    // Compute dJ / dv(r) according to activation function
    f32 v_k_derivative = transform_yk_derivative_to_vk_derivative(y, y_k_derivarive, v_gradient, activation_func);

    // Store derivatives in output vector
    v_gradient.vals[k] = v_k_derivative;
}


__global__ void compute_output_layer_v_gradient(vector_t y_out, vector_t y_out_expected, vector_t v_out_gradient, activation_func_t activation_func, cost_func_t cost_func) {
    u32 k = threadIdx.x;

    // Compute gradient of y_out
    f32 y_k_derivarive = compute_output_layer_y_gradient(y_out, y_out_expected, cost_func);

    // Compute gradient of v_out
    f32 v_k_derivative = transform_yk_derivative_to_vk_derivative(y_out, y_k_derivarive, v_out_gradient, activation_func);

    // Store gradient of v_out
    v_out_gradient.vals[k] = v_k_derivative;
}


// Call with
// gridDim.x = layer.out_dim
// blockDim.x = layer.in_dim
__global__ void compute_weight_gradient(matrix_t w_derivative_out, vector_t v_gradient, vector_t y_minus) {
    u32 i = blockIdx.x;
    u32 j = gridDim.x;
    f32 v_gradient_i = v_gradient.vals[i];
    f32 y_minus_j = y_minus.vals[j];
    *matrix_index(w_derivative_out, i, j) = v_gradient_i * y_minus_j;
}
