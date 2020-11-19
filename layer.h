#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "linalg.h"
#include "activation_func.h"


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
} layer_builder_t;


void layer_init(layer_t *layer, layer_builder_t *layer_builder, f32 **param_buffer);

__global__ void layer_compute(layer_t layer, device_vector_t in_vector, device_vector_t out_vector);

#endif
