#include "csv.h"
#include "mlp.h"
#include "common.h"
#include "linalg.h"


int main() {

    // Create MLP with pre-initialized parameters
    mlp_builder_t builder = mlp_builder_create(2);
    mlp_builder_add_layer(&builder, 3, SIGMOID, 0.005);
    mlp_builder_add_layer(&builder, 2, SOFTMAX, 0.05);
    mlp_t mlp = mlp_builder_finalize(&builder);
    f32 params[] = {
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        1.0, -1.0, 0.5,
        1.0, 1.0, -1.0, 1.0, 1.0, 0.0,
        -1.0, -1.0
    };
    int x = cudaMemcpy(mlp.all_parameters, params, sizeof(params), cudaMemcpyHostToDevice);

    // Alloc gradient
    mlp_gradient_compute_data_t gradient;
    gradient = mlp_alloc_gradient_compute_data(mlp);

    // Prepare input vector, and expected-output vector
    device_vector_t device_input, device_output;
    vector_init(&device_input, 2);
    vector_init(&device_output, 2);
    f32 output_data[2] = {0, 1};
    f32 input_data[2] = {20, 30};
    host_vector_t output = {
        .size = 2,
        .vals = output_data
    };
    host_vector_t input = {
        .size = 2,
        .vals = input_data
    };

    // Compute gradient, and update parameters in MLP
    f32 STEP_SIZE = 0.1;
    gradient_descent_update_host_vectors(mlp, gradient, input, output,
            device_input, device_output, CROSS_ENTROPY, STEP_SIZE);

    // Print gradient
    printf("\n\nGradients\n");
    for (u32 i = 0; i != 2; i++) {
        layer_params_t params = gradient.gradient[i];
        host_print_matrix(params.weights, (char *) "weights");
        host_print_vector(params.bias, (char *) "bias");
    }

    // Print new parameters
    printf("\n\nNew parameters\n");
    for (u32 i = 0; i != 2; i++) {
        layer_params_t params = mlp.layers[i].params;
        host_print_matrix(params.weights, (char *) "weights");
        host_print_vector(params.bias, (char *) "bias");
    }

    // Verify the gradient
    f32 G = 0.268941;
    ASSERT_DEVICE_VECTOR(gradient.gradient[0].bias, 3, 0.0, 0.0, 0.0);
    ASSERT_DEVICE_MATRIX(gradient.gradient[0].weights, 3, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    ASSERT_DEVICE_VECTOR(gradient.gradient[1].bias, 2, -G, G);
    ASSERT_DEVICE_MATRIX(gradient.gradient[1].weights, 2, 3, -G, -G, -G, G, G, G);

    // Verify the new parameters
    f32 S = -G * STEP_SIZE;
    ASSERT_DEVICE_VECTOR(mlp.layers[0].params.bias, 3, 1.0f, -1.0f, 0.5f);
    ASSERT_DEVICE_MATRIX(mlp.layers[0].params.weights, 3, 2, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f);
    ASSERT_DEVICE_VECTOR(mlp.layers[1].params.bias, 2, -1.0f-S, -1.0f + S);
    ASSERT_DEVICE_MATRIX(mlp.layers[1].params.weights, 2, 3, 1.0f-S, 1.0f-S, -1.0f-S, 1.0f+S, 1.0f+S, S);
}