#include "csv.h"
#include "mlp.h"
#include "common.h"
#include "linalg.h"


void encode_one_hot(f32 *arr, u32 size, u32 label) {
    for (u32 i = 0; i != size; i++) {
        arr[i] = 0.0;
    }
    arr[label] = 1.0;
}


int main() {

    // Parse data
    csv_parser_t parser = csv_parser_create((char *) "test_data/mnist_train.csv", ',', 785);
    u32 num_vecs;
    f32 **rows = csv_parser_collect(&parser, &num_vecs);
    ASSERT_EQ(num_vecs, 60000);

    // Create MLP
    mlp_builder_t builder = mlp_builder_create(784);
    mlp_builder_add_layer(&builder, 300, SIGMOID, 0.00005);
    mlp_builder_add_layer(&builder, 10, SOFTMAX, 0.0005);
    mlp_t mlp = mlp_builder_finalize(&builder);
    mlp_gradient_compute_data_t gradient;
    gradient = mlp_alloc_gradient_compute_data(mlp);
    device_vector_t device_input, device_output;
    vector_init(&device_input, 784);
    vector_init(&device_output, 10);

    // Train MLP
    for (u32 epoch = 0; epoch != 2; epoch++) {
        f32 step_size = -0.001f;
        printf("epoch %d, step_size: %f\n", epoch, step_size);
        for (u32 i = 0; i != num_vecs; i++) {
            if (i % 500 == 0) {
                printf("epoch %d. datapoint %d / %d\n", epoch, i, num_vecs);
            }
            f32 *row = rows[i];
            u32 label = (u32) row[0];
            f32 output_data[10];
            encode_one_hot(output_data, 10, label);
            host_vector_t output = {
                .size = 10,
                .vals = output_data
            };
            host_vector_t input = {
                .size = 784,
                .vals = &row[1]
            };

            gradient_descent_update_host_vectors(mlp, gradient, input, output,
                    device_input, device_output, CROSS_ENTROPY, step_size);
        }
    }

    // Print output of the first 100 vectors
    for (u32 i = 0; i != 100; i++) {
        // Compute output from input
        f32 *row = rows[i];
        host_vector_t input = {
            .size = 784,
            .vals = &row[1]
        };
        vector_host_to_device(&device_input, &input);
        network_compute(mlp, gradient.layer_outputs, device_input);

        // Print output
        printf("\nExpected output %d\n", (i32) row[0]);
        host_print_vector(gradient.layer_outputs[1], (char *) "output");
    }
}

