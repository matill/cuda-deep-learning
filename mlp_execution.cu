#include "mlp.h"
#include "linalg.h"
#include "layer.h"
#include "common.h"
#include <stdio.h>


void network_compute(mlp_t mlp, device_vector_t *layer_outputs, device_vector_t in_vector) {
    // Compute output of each layer
    for (int i = 0; i != mlp.num_layers; i++) {
        device_vector_t layer_in_vec = (i == 0) ? in_vector : layer_outputs[i-1];
        device_vector_t layer_out_vec = layer_outputs[i];
        layer_t layer = mlp.layers[i];
        layer_compute<<<1, layer.out_dimension>>>(layer, layer_in_vec, layer_out_vec);
        cudaDeviceSynchronize();
    }
}


// mlp: host-located. mlp.layers is also host-located data
// gradient: host-located
// layer_outputs: host-located array of vectors, where vector-data is on device
// in_vector: host-located vector where data is on device
// expected_out_vector: host-located vector where data is on device
void compute_gradient(mlp_t mlp, mlp_gradient_compute_data_t gradient_compute_data,
        device_vector_t input, device_vector_t output_truth, cost_func_t cost_func) {

    ASSERT_EQ(mlp.layers[0].in_dimension, input.size)
    ASSERT_EQ(mlp.layers[mlp.num_layers-1].out_dimension, output_truth.size)

    layer_params_t *gradient = gradient_compute_data.gradient;
    device_vector_t *layer_outputs = gradient_compute_data.layer_outputs;

    // Forward computation
    network_compute(mlp, layer_outputs, input);

    // Backward computation
    i32 last_layer = mlp.num_layers - 1;
    for (i32 i = last_layer; i != -1; i--) {
        layer_params_t layer_gradient = gradient[i];
        layer_t layer = mlp.layers[i];

        // Compute v gradient (= bias gradient)
        u32 in_dim = layer.in_dimension;
        u32 out_dim = layer.out_dimension;
        device_vector_t layer_output = layer_outputs[i];
        activation_func_t activation_func = layer.activation_func;
        device_vector_t bias_and_v_gradient = layer_gradient.bias;
        if (i == last_layer) {
            compute_output_layer_v_gradient<<<1, out_dim>>>(layer_output, output_truth, bias_and_v_gradient, activation_func, cost_func);
        } else {
            device_vector_t v_plus_gradient = gradient[i+1].bias;
            device_matrix_t w_plus = mlp.layers[i+1].params.weights;
            compute_v_gradient_from_v_plus_gradient<<<1, out_dim>>>(bias_and_v_gradient, v_plus_gradient, w_plus, layer_output, activation_func);
        }
        cudaDeviceSynchronize();

        // Compute weight gradient
        device_vector_t y_minus = i == 0 ? input : layer_outputs[i-1];
        device_matrix_t weight_gradient = layer_gradient.weights;
        compute_weight_gradient<<<out_dim, in_dim>>>(weight_gradient, bias_and_v_gradient, y_minus);
    }
    cudaDeviceSynchronize();
}


__global__ void update_params(f32 *params, f32 *gradient, f32 step_size, u32 num_params) {
    u32 start_index = threadIdx.x + blockIdx.x * blockDim.x;
    u32 num_threads = blockDim.x * gridDim.x;

    for (u32 i = start_index; i < num_params; i += num_threads) {
        f32 step = gradient[i] * step_size;
        params[i] -= step;
    }
}


void gradient_descent_update_device_vectors(mlp_t mlp, mlp_gradient_compute_data gradient_compute_data,
    device_vector_t input, device_vector_t output_truth, cost_func_t cost_func, f32 step_size) {

    ASSERT_EQ(mlp.layers[0].in_dimension, input.size)
    ASSERT_EQ(mlp.layers[mlp.num_layers-1].out_dimension, output_truth.size)

    compute_gradient(mlp, gradient_compute_data, input, output_truth, cost_func);

    // Update MLP according to gradient
    u32 num_params = mlp.num_parameters;
    u32 num_threads, num_blocks;
    if (num_params > 512) {
        num_threads = 512;
        num_blocks = num_params / 512;
    } else {
        num_threads = num_params;
        num_blocks = 1;
    }
    update_params<<<num_blocks, num_threads>>>(mlp.all_parameters, gradient_compute_data.all_parameters,
            step_size, num_params);
    cudaDeviceSynchronize();
}


void gradient_descent_update_host_vectors(mlp_t mlp, mlp_gradient_compute_data gradient_compute_data,
        host_vector_t input, host_vector_t output_truth, device_vector_t device_in_buf,
        device_vector_t device_out_buf, cost_func_t cost_func, f32 step_size) {

    ASSERT_EQ(mlp.layers[0].in_dimension, device_in_buf.size)
    ASSERT_EQ(mlp.layers[mlp.num_layers-1].out_dimension, device_out_buf.size)
    ASSERT_EQ(input.size, device_in_buf.size)
    ASSERT_EQ(output_truth.size, device_out_buf.size)
    vector_host_to_device(&device_in_buf, &input);
    vector_host_to_device(&device_out_buf, &output_truth);
    cudaDeviceSynchronize();
    gradient_descent_update_device_vectors(mlp, gradient_compute_data, device_in_buf,
            device_out_buf, cost_func, step_size);
}

