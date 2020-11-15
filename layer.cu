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

