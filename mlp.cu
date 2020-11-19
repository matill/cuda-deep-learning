#include "mlp.h"
#include "linalg.h"
#include "layer.h"
#include "common.h"
#include <stdio.h>


void network_compute(mlp_t *mlp, device_vector_t *layer_outputs, device_vector_t *in_vector) {
    // Compute output of each layer
    for (int i = 0; i != mlp->num_layers; i++) {
        device_vector_t *layer_in_vec = (i == 0) ? in_vector : &layer_outputs[i-1];
        device_vector_t *layer_out_vec = &layer_outputs[i];
        layer_t *layer = &mlp->layers[i];
        layer_compute<<<1, layer->out_dimension>>>(*layer, *layer_in_vec, *layer_out_vec);
        cudaDeviceSynchronize();
    }
}


// mlp: host-located. mlp.layers is also host-located data
// gradient: host-located
// layer_outputs: host-located array of vectors, where vector-data is on device
// in_vector: host-located vector where data is on device
// expected_out_vector: host-located vector where data is on device
void compute_gradient(mlp_t *mlp, mlp_t *gradient, device_vector_t *layer_outputs, device_vector_t *in_vector, device_vector_t *expected_out_vector) {
    network_compute(mlp, layer_outputs, in_vector);
    // Compute derivative with respect to output layer dJ/dy(L)
    // vector_t *mlp_output = layer_outputs[mlp->num_layers];
    // cost_function_derivative(mlp->cost_func, mlp_output, expected_out_vector, )
}


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

    // Alloc array of layer descriptors
    mlp_t mlp = {
        .num_layers = mlp_builder->num_layers,
        .layers = (layer_t *) malloc(sizeof(layer_t) * mlp_builder->num_layers)
    };
    ASSERT_NOT_NULL(mlp.layers);

    // Alloc continous device memory for weights and bias.
    u32 num_floats = 0;
    for (u32 i = 0; i != mlp_builder->num_layers; i++) {
        layer_builder_t layer_builder = mlp_builder->layers[i];
        num_floats += layer_builder.out_dimension * (layer_builder.in_dimension + 1);
    }

    f32 *param_buffer;
    cudaMalloc(&param_buffer, sizeof(f32) * num_floats);
    ASSERT_NOT_NULL(param_buffer);

    // Initialize layers
    for (u32 i = 0; i != mlp.num_layers; i++) {
        layer_builder_t *layer_builder = &mlp_builder->layers[i];
        layer_t *layer = &mlp.layers[i];
        layer_init(layer, layer_builder, &param_buffer);
    }

    return mlp;
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