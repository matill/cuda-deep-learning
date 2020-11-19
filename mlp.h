#ifndef MLP_H
#define MLP_H

#include "common.h"
#include "layer.h"
#include "cost_func.h"


typedef struct mlp {
    u32 num_layers;
    layer_t *layers;
} mlp_t;


typedef struct mlp_builder {
    u32 num_layers;
    u32 in_dimension;
    layer_builder_t layers[100];
} mlp_builder_t;


typedef struct mlp_gradient_compute_data {
    layer_params_t *gradient;
    device_vector_t *layer_outputs;
} mlp_gradient_compute_data_t;

// mlp_trainer_t contains a multilayer perceptron, and the datastructures/memory
// that the mlp needs to perform gradient descent updates.
typedef struct mlp_trainer {
    mlp_t mlp;
    mlp_t gradient;
} mlp_trainer_t;


void network_compute(mlp_t *mlp, vector_t *layer_outputs, vector_t *in_vector);

mlp_builder_t mlp_builder_create(u32 in_dimension);
void mlp_builder_add_layer(mlp_builder_t *mlp_builder, u32 out_dimension, activation_func_t activation_func);
mlp_t mlp_builder_finalize(mlp_builder_t *mlp_builder);

#endif
