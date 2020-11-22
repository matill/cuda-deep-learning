#include "layer.h"


void layer_init(layer_t *layer, layer_builder_t *layer_builder, f32 **param_buffer) {
    layer->in_dimension = layer_builder->in_dimension;
    layer->out_dimension = layer_builder->out_dimension;
    layer->activation_func = layer_builder->activation_func;
    u32 height = layer->out_dimension;
    u32 width = layer->in_dimension;
    matrix_init_from_buf(&layer->params.weights, height, width, param_buffer);
    vector_init_from_buf(&layer->params.bias, height, param_buffer);
}


__global__ void layer_compute(layer_t layer, device_vector_t in_vector, device_vector_t out_vector) {
    ASSERT_EQ_INT(layer.in_dimension, in_vector.size);
    ASSERT_EQ_INT(layer.out_dimension, out_vector.size);
    device_matrix_t weights = layer.params.weights;
    device_vector_t bias = layer.params.bias;

    // out_vector = weights * in_vector + bias
    matrix_vector_multiply(weights, in_vector, out_vector);
    out_vector.vals[threadIdx.x] += bias.vals[threadIdx.x];

    apply_activation_func(out_vector, layer.activation_func);
}



__global__ void compute_v_gradient_from_v_plus_gradient(device_vector_t v_gradient,
    device_vector_t v_plus_gradient, device_matrix_t w_plus, device_vector_t y,
    activation_func_t activation_func) {

    // Compute dJ / dy(r)
    u32 k = threadIdx.x;
    f32 y_k_derivarive = 0;
    u32 num_iters = w_plus.height;
    for (u32 i = 0; i != num_iters; i++) {
        y_k_derivarive += v_plus_gradient.vals[i] * *matrix_index(w_plus, i, k);
    }

    // Compute dJ / dv(r) according to activation function
    f32 v_k_derivative = transform_yk_derivative_to_vk_derivative(y, y_k_derivarive, v_gradient, activation_func);

    // Store derivatives in output vector
    v_gradient.vals[k] = v_k_derivative;
}


__global__ void compute_output_layer_v_gradient(device_vector_t y_out,
    device_vector_t y_out_expected, device_vector_t v_out_gradient,
    activation_func_t activation_func, cost_func_t cost_func) {

    u32 k = threadIdx.x;

    // Compute gradient of y_out
    f32 y_k_derivarive = compute_output_layer_y_gradient(y_out, y_out_expected, cost_func);

    // Compute gradient of v_out
    f32 v_k_derivative = transform_yk_derivative_to_vk_derivative(y_out, y_k_derivarive, v_out_gradient, activation_func);

    // Store gradient of v_out
    v_out_gradient.vals[k] = v_k_derivative;
}


// Call with <<<layer.out_dim, layer.in_dim>>>
__global__ void compute_weight_gradient(device_matrix_t w_derivative_out,
    device_vector_t v_gradient, device_vector_t y_minus) {

    u32 i = blockIdx.x;
    u32 j = gridDim.x;
    f32 v_gradient_i = v_gradient.vals[i];
    f32 y_minus_j = y_minus.vals[j];
    *matrix_index(w_derivative_out, i, j) = v_gradient_i * y_minus_j;
}
