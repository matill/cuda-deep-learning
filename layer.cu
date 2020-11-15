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
