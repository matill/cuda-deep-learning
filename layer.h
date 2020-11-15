#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "linalg.h"
#include "activation_func.h"


typedef struct layer {
    u32 in_dimension;
    u32 out_dimension;
    activation_func_t activation_func;
    matrix_t weights;
    vector_t bias;
} layer_t;


void layer_init(layer_t *layer, u32 in_dimension, u32 out_dimension, activation_func_t activation_func);


#endif
