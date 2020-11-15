#include "layer.h"
#include "common.h"
#include "linalg.h"


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


// TODO: Only accept subsets of the structs
__global__ void transform_gradient_y_to_v_softmax(vector_t y_gradient, vector_t y) {
    f32 sigma_c = vector_dot(y_gradient, y);
    __syncthreads();
    u32 k = threadIdx.x;
    f32 y_k = y.vals[k];
    f32 *y_k_derivative = &y_gradient.vals[k];
    *y_k_derivative = y_k * (*y_k_derivative  - sigma_c);
}


// TODO: Only accept subsets of the structs
__global__ void transform_gradient_y_to_v_sigmoid(vector_t y_gradient, vector_t y) {
    u32 k = threadIdx.x;
    f32 y_k = y.vals[k];
    y_gradient.vals[k] = y_k * (y_k - 1) * y_gradient.vals[k];
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
