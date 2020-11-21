#include "mlp.h"
#include "linalg.h"
#include "layer.h"
#include "common.h"
#include <stdio.h>


device_vector_t *alloc_layer_outputs(mlp_t *mlp) {
    device_vector_t *vectors = (device_vector_t *) malloc(sizeof(device_vector_t) * mlp->num_layers);
    ASSERT_NOT_NULL(vectors);
    for (u32 i = 0; i != mlp->num_layers; i++) {
        vector_init(&vectors[i], mlp->layers[i].out_dimension);
    }

    return vectors;
}


mlp_builder_t mlp_builder_create(u32 in_dimension) {
    mlp_builder_t builder = {
        .num_layers = 0,
        .in_dimension = in_dimension
    };
    return builder;
}


void mlp_builder_add_layer(mlp_builder_t *mlp_builder, u32 out_dimension, activation_func_t activation_func, f32 rand_range) {
    u32 num_layers = mlp_builder->num_layers;
    u32 in_dimension = num_layers == 0 ? mlp_builder->in_dimension : mlp_builder->layers[num_layers-1].out_dimension;
    layer_builder_t layer_builder = {
        .in_dimension = in_dimension,
        .out_dimension = out_dimension,
        .activation_func = activation_func,
        .rand_range = rand_range
    };

    mlp_builder->layers[mlp_builder->num_layers++] = layer_builder;
}


mlp_t mlp_builder_finalize(mlp_builder_t *mlp_builder) {
    ASSERT_NEQ_INT(mlp_builder->num_layers, 0);

    // Find total number of parameters in the mlp
    u32 num_parameters = 0;
    for (u32 i = 0; i != mlp_builder->num_layers; i++) {
        layer_builder_t layer_builder = mlp_builder->layers[i];
        num_parameters += layer_builder.out_dimension * (layer_builder.in_dimension + 1);
    }

    // Alloc memory for all parameters in the mlp (weights and bias)
    f32 *all_parameters;
    u32 x = cudaMalloc(&all_parameters, sizeof(f32) * num_parameters);
    ASSERT_EQ(x, 0);
    ASSERT_NOT_NULL(all_parameters);

    // Initialize layers
    f32 *parameter_data_iter = all_parameters;
    layer_t *layers = (layer_t *) malloc(sizeof(layer_t) * mlp_builder->num_layers);
    ASSERT_NOT_NULL(layers);
    for (u32 i = 0; i != mlp_builder->num_layers; i++) {
        layer_t *layer = &layers[i];
        layer_builder_t *layer_builder = &mlp_builder->layers[i];
        f32 rand_range = layer_builder->rand_range;
        layer_init(layer, layer_builder, &parameter_data_iter);
        vector_set_rand_unif_vals(&layer->params.bias, -rand_range, rand_range);
        matrix_set_rand_unif_vals(&layer->params.weights, -rand_range, rand_range);
    }

    // Return as struct
    mlp_t mlp = {
        .num_layers = mlp_builder->num_layers,
        .layers = layers,
        .num_parameters = num_parameters,
        .all_parameters = all_parameters
    };
    return mlp;
}


mlp_gradient_compute_data_t mlp_alloc_gradient_compute_data(mlp_t mlp) {

    // Alloc continous device memory for gradient of weights and bias
    f32 *all_parameters;
    cudaMalloc(&all_parameters, sizeof(f32) * mlp.num_parameters);
    ASSERT_NOT_NULL(all_parameters);

    // Initialize gradients
    f32 *parameter_data_iter = all_parameters;
    layer_params_t *gradient = (layer_params_t *) malloc(sizeof(layer_params_t) * mlp.num_layers);
    ASSERT_NOT_NULL(gradient)
    for (u32 i = 0; i != mlp.num_layers; i++) {
        layer_t *layer = &mlp.layers[i];
        u32 height = layer->out_dimension;
        u32 width = layer->in_dimension;
        matrix_init_from_buf(&gradient[i].weights, height, width, &parameter_data_iter);
        vector_init_from_buf(&gradient[i].bias, height, &parameter_data_iter);
    }

    // Alloc data for layer outputs
    device_vector_t *layer_outputs = (device_vector_t *) malloc(sizeof(device_vector_t) * mlp.num_layers);
    ASSERT_NOT_NULL(layer_outputs);
    for (u32 i = 0; i != mlp.num_layers; i++) {
        device_vector_t *vector = &layer_outputs[i];
        vector_init(vector, mlp.layers[i].out_dimension);
    }

    // Return result
    mlp_gradient_compute_data_t result = {
        .gradient = gradient,
        .layer_outputs = layer_outputs,
        .all_parameters = all_parameters
    };
    return result;
}

