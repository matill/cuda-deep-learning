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


void mlp_builder_add_layer(mlp_builder_t *mlp_builder, u32 out_dimension, activation_func_t activation_func) {
    u32 num_layers = mlp_builder->num_layers;
    u32 in_dimension = num_layers == 0 ? mlp_builder->in_dimension : mlp_builder->layers[num_layers-1].out_dimension;
    layer_builder_t layer_builder = {
        .in_dimension = in_dimension,
        .out_dimension = out_dimension,
        .activation_func = activation_func
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
    cudaMalloc(&all_parameters, sizeof(f32) * num_parameters);
    ASSERT_NOT_NULL(all_parameters);

    // Initialize layers
    f32 *parameter_data_iter = all_parameters;
    layer_t *layers = (layer_t *) malloc(sizeof(layer_t) * mlp_builder->num_layers);
    ASSERT_NOT_NULL(layers);
    for (u32 i = 0; i != mlp_builder->num_layers; i++) {
        layer_init(&layers[i], &mlp_builder->layers[i], &parameter_data_iter);
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


int main() {

    // // Values in layers
    // f32 layer_0_bias_raw[5] = {
    //     -2, -1, 0, 1, 2
    // };
    // f32 layer_1_bias_raw[3] = {
    //     2, 2, 2
    // };
    // host_vector_t layer_0_bias = {
    //     .size = 5,
    //     .vals = layer_0_bias_raw
    // };
    // host_vector_t layer_1_bias = {
    //     .size = 3,
    //     .vals = layer_1_bias_raw
    // };

    // mlp_t mlp = {
    //     .num_layers = 2,
    //     .layers = (layer_t *) malloc(sizeof(layer_t) * 2),
    // };

    // // Create input vector and mlp
    // device_vector_t input;
    // vector_init(&input, 2);
    // layer_init(&mlp.layers[0], 2, 5, SIGMOID);
    // layer_init(&mlp.layers[1], 5, 3, SOFTMAX);
    // device_vector_t *layer_outputs = alloc_layer_outputs(&mlp);

    // // Move bias vectors into MLP layers
    // vector_host_to_device(&mlp.layers[0].bias, &layer_0_bias);
    // vector_host_to_device(&mlp.layers[1].bias, &layer_1_bias);

    // // Compute
    // network_compute(&mlp, layer_outputs, &input);

    // // Alloc memory to get data from device
    // host_vector_t host_layer_outputs[2] = {
    //     {
    //         .size = 5,
    //         .vals = (f32 *) malloc(sizeof(f32) * 10)
    //     },
    //     {
    //         .size = 3,
    //         .vals = (f32 *) malloc(sizeof(f32) * 10)
    //     }
    // };

    // for (i32 i = 0; i != 10; i++) {
    //     host_layer_outputs[0].vals[i] = -2*i;
    //     host_layer_outputs[1].vals[i] = -2*i;
    // }

    // vector_device_to_host(&host_layer_outputs[0], &layer_outputs[0]);
    // for (i32 i = 0; i != 10; i++)
    //     printf("layers 0 [%d] = %f\n", i, host_layer_outputs[0].vals[i]);

    // vector_device_to_host(&host_layer_outputs[1], &layer_outputs[1]);
    // for (i32 i = 0; i != 10; i++)
    //     printf("layers 1 [%d] = %f\n", i, host_layer_outputs[1].vals[i]);


}


// __global__ void cross_entropy_derivative(device_vector_t *estimate, device_vector_t *truth, device_vector_t *derivative_out) {

// }




// __global__ void layer_


// __global__ void layer_back_propagate_step(layer_t *layer, device_vector_t *y_self, device_vector_t *y_minus, )