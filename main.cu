#include "csv.h"
#include "mlp.h"
#include "common.h"
#include "linalg.h"


int main() {

    // Parse data
    csv_parser_t parser = csv_parser_create((char *) "test_data/linearly_separable.csv", ',', 3);
    u32 num_vecs;
    f32 **rows = csv_parser_collect(&parser, &num_vecs);
    ASSERT_EQ(num_vecs, 8);
    for (u32 i = 0; i != num_vecs; i++) {
        f32 *row = rows[i];
        printf("rows[%d] = [%.1f, %.1f, %.1f]\n", i, row[0], row[1], row[2]);
    }

    // Create MLP
    mlp_builder_t builder = mlp_builder_create(2);
    mlp_builder_add_layer(&builder, 2, SOFTMAX, 0.05);
    mlp_t mlp = mlp_builder_finalize(&builder);
    mlp_gradient_compute_data_t gradient;
    gradient = mlp_alloc_gradient_compute_data(mlp);
    device_vector_t device_input, device_output;
    vector_init(&device_input, 2);
    vector_init(&device_output, 1);

    // Train MLP
    for (u32 epoch = 0; epoch != 10; epoch++) {
        for (u32 i = 0; i != num_vecs; i++) {
            f32 *row = rows[i];
            i32 label = (i32) row[0];
            printf("label: %d\n", label);
            f32 output_data[2] = {0, 0};
            output_data[label] = 1;
            host_vector_t output = {
                .size = 2,
                .vals = output_data
            };
            host_vector_t input = {
                .size = 2,
                .vals = &row[1]
            };

            gradient_descent_update_host_vectors(mlp, gradient, input, output,
                    device_input, device_output, CROSS_ENTROPY, 0.001);

            printf("epoch: %d. i: %d\n", epoch, i);
        }
    }

    // Print output of MLP
    for (u32 i = 0; i != num_vecs; i++) {

        // Compute output from input
        f32 *row = rows[i];
        host_vector_t input = {
            .size = 2,
            .vals = &row[1]
        };
        vector_host_to_device(device_input, input);
        network_compute(&mlp, gradient.layer_outputs, &input);

        // Copy output to host memory
        f32 output_data[2];
        host_vector_t output = {
            .size = 2,
            .vals = output_data
        };

        vector_device_to_host(output, gradient.layer_outputs[0]);

        // Print output
        printf("output %d: [%.1f, %.1f]\n", output_data[0], output_data[1]);
    }
}

