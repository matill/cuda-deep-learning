#ifndef MLP_H
#define MLP_H

#include "common.h"
#include "layer.h"
#include "cost_func.h"


typedef struct multilayer_perceptron {
    u32 num_layers;
    layer_t *layers;
    cost_func_t cost_func;
} mlp_t;


typedef struct mlp_gradient_compute {
    mlp_t gradient;
    vector_t *layer_outputs;
} mlp_gradient_compute_t;

// mlp_trainer_t contains a multilayer perceptron, and the datastructures/memory
// that the mlp needs to perform gradient descent updates.
typedef struct mlp_trainer {
    mlp_t mlp;
    mlp_t gradient;
} mlp_trainer_t;


void network_compute(mlp_t *mlp, vector_t *layer_outputs, vector_t *in_vector);


#endif
