#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "linalg.h"
#include "activation_func.h"
#include "cost_func.h"


typedef struct layer_params {
    device_matrix_t weights;
    device_vector_t bias;
} layer_params_t;


typedef struct layer {
    u32 in_dimension;
    u32 out_dimension;
    activation_func_t activation_func;
    layer_params_t params;
} layer_t;


typedef struct layer_builder {
    u32 in_dimension;
    u32 out_dimension;
    activation_func_t activation_func;
    f32 rand_range;
} layer_builder_t;


void layer_init(layer_t *layer, layer_builder_t *layer_builder, f32 **param_buffer);


__global__ void layer_compute(layer_t layer, device_vector_t in_vector, device_vector_t out_vector);


__global__ void compute_v_gradient_from_v_plus_gradient(device_vector_t v_gradient,
        device_vector_t v_plus_gradient, device_matrix_t w_plus, device_vector_t y,
        activation_func_t activation_func);


__global__ void compute_output_layer_v_gradient(device_vector_t y_out,
        device_vector_t y_out_expected, device_vector_t v_out_gradient, 
        activation_func_t activation_func, cost_func_t cost_func);


// Call with
// gridDim.x = layer.out_dim
// blockDim.x = layer.in_dim
__global__ void compute_weight_gradient(device_matrix_t w_derivative_out,
        device_vector_t v_gradient, device_vector_t y_minus);


#endif

