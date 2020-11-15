#include "mlp.h"
#include "linalg.h"
#include "layer.h"
#include "common.h"
#include <stdio.h>


void network_compute(mlp_t *mlp, vector_t *layer_outputs, vector_t *in_vector) {
    // Compute output of each layer
    for (int i = 0; i != mlp->num_layers; i++) {
        vector_t *layer_in_vec = (i == 0) ? in_vector : &layer_outputs[i-1];
        vector_t *layer_out_vec = &layer_outputs[i];
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
void compute_gradient(mlp_t *mlp, mlp_t *gradient, vector_t *layer_outputs, vector_t *in_vector, vector_t *expected_out_vector) {
    network_compute(mlp, layer_outputs, in_vector);
    // Compute derivative with respect to output layer dJ/dy(L)
    // vector_t *mlp_output = layer_outputs[mlp->num_layers];
    // cost_function_derivative(mlp->cost_func, mlp_output, expected_out_vector, )
}


vector_t *alloc_layer_outputs(mlp_t *mlp) {
    vector_t *vectors = (vector_t *) malloc(sizeof(vector_t) * mlp->num_layers);
    ASSERT_NOT_NULL(vectors);
    for (u32 i = 0; i != mlp->num_layers; i++) {
        vector_init(&vectors[i], mlp->layers[i].out_dimension);
    }

    return vectors;
}


int main() {

    // Values in layers
    f32 layer_0_bias_raw[5] = {
        -2, -1, 0, 1, 2
    };
    f32 layer_1_bias_raw[3] = {
        2, 2, 2
    };
    vector_t layer_0_bias = {
        .size = 5,
        .vals = layer_0_bias_raw
    };
    vector_t layer_1_bias = {
        .size = 3,
        .vals = layer_1_bias_raw
    };

    mlp_t mlp = {
        .num_layers = 2,
        .layers = (layer_t *) malloc(sizeof(layer_t) * 2),
        .cost_func = CROSS_ENTROPY
    };

    // Create input vector and mlp
    vector_t input;
    vector_init(&input, 2);
    layer_init(&mlp.layers[0], 2, 5, SIGMOID);
    layer_init(&mlp.layers[1], 5, 3, SOFTMAX);
    vector_t *layer_outputs = alloc_layer_outputs(&mlp);

    // Move bias vectors into MLP layers
    vector_host_to_device(&mlp.layers[0].bias, &layer_0_bias);
    vector_host_to_device(&mlp.layers[1].bias, &layer_1_bias);

    // Compute
    network_compute(&mlp, layer_outputs, &input);

    // Alloc memory to get data from device
    vector_t host_layer_outputs[2] = {
        {
            .size = 5,
            .vals = (f32 *) malloc(sizeof(f32) * 10)
        },
        {
            .size = 3,
            .vals = (f32 *) malloc(sizeof(f32) * 10)
        }
    };

    for (i32 i = 0; i != 10; i++) {
        host_layer_outputs[0].vals[i] = -2*i;
        host_layer_outputs[1].vals[i] = -2*i;
    }

    vector_device_to_host(&host_layer_outputs[0], &layer_outputs[0]);
    for (i32 i = 0; i != 10; i++)
        printf("layers 0 [%d] = %f\n", i, host_layer_outputs[0].vals[i]);

    vector_device_to_host(&host_layer_outputs[1], &layer_outputs[1]);
    for (i32 i = 0; i != 10; i++)
        printf("layers 1 [%d] = %f\n", i, host_layer_outputs[1].vals[i]);


}


// __global__ void cross_entropy_derivative(vector_t *estimate, vector_t *truth, vector_t *derivative_out) {

// }




// __global__ void layer_


// __global__ void layer_back_propagate_step(layer_t *layer, vector_t *y_self, vector_t *y_minus, )