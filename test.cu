#include "mlp.h"
#include <stdio.h>


void test_create_mlp_builder(void) {
    // Test with 1 layer
    mlp_builder_t mlp_builder = mlp_builder_create(6);
    mlp_builder_add_layer(&mlp_builder, 5, SIGMOID, 0.10);
    ASSERT_EQ(mlp_builder.num_layers, 1)
    ASSERT_EQ(mlp_builder.in_dimension, 6)
    ASSERT_EQ(mlp_builder.layers[0].in_dimension, 6);
    ASSERT_EQ(mlp_builder.layers[0].out_dimension, 5);
    ASSERT_EQ(mlp_builder.layers[0].activation_func, SIGMOID);
    ASSERT_EQ(mlp_builder.layers[0].rand_range, 0.1f);

    // Test with 2 layers
    mlp_builder_add_layer(&mlp_builder, 2, SOFTMAX, 0.20);
    ASSERT_EQ(mlp_builder.num_layers, 2)
    ASSERT_EQ(mlp_builder.in_dimension, 6)

    ASSERT_EQ(mlp_builder.layers[0].in_dimension, 6);
    ASSERT_EQ(mlp_builder.layers[0].out_dimension, 5);
    ASSERT_EQ(mlp_builder.layers[0].activation_func, SIGMOID);
    ASSERT_EQ(mlp_builder.layers[0].rand_range, 0.1f);

    ASSERT_EQ(mlp_builder.layers[1].in_dimension, 5);
    ASSERT_EQ(mlp_builder.layers[1].out_dimension, 2);
    ASSERT_EQ(mlp_builder.layers[1].activation_func, SOFTMAX);
    ASSERT_EQ(mlp_builder.layers[1].rand_range, 0.2f);

    // Test with 3 layers
    mlp_builder_add_layer(&mlp_builder, 1, SIGMOID, 0.15);
    ASSERT_EQ(mlp_builder.num_layers, 3)
    ASSERT_EQ(mlp_builder.in_dimension, 6)

    ASSERT_EQ(mlp_builder.layers[0].in_dimension, 6);
    ASSERT_EQ(mlp_builder.layers[0].out_dimension, 5);
    ASSERT_EQ(mlp_builder.layers[0].activation_func, SIGMOID);
    ASSERT_EQ(mlp_builder.layers[0].rand_range, 0.1f);

    ASSERT_EQ(mlp_builder.layers[1].in_dimension, 5);
    ASSERT_EQ(mlp_builder.layers[1].out_dimension, 2);
    ASSERT_EQ(mlp_builder.layers[1].activation_func, SOFTMAX);
    ASSERT_EQ(mlp_builder.layers[1].rand_range, 0.2f);

    ASSERT_EQ(mlp_builder.layers[2].in_dimension, 2);
    ASSERT_EQ(mlp_builder.layers[2].out_dimension, 1);
    ASSERT_EQ(mlp_builder.layers[2].activation_func, SIGMOID);
    ASSERT_EQ(mlp_builder.layers[2].rand_range, 0.15f);
}


void test_finalize_mlp_builder(void) {
    // Test with 1 layer
    mlp_builder_t builder = mlp_builder_create(10);
    mlp_builder_add_layer(&builder, 9, SIGMOID, 0.1);
    mlp_t mlp = mlp_builder_finalize(&builder);

    ASSERT_EQ(mlp.num_layers, 1)
    ASSERT_EQ(mlp.num_parameters, 10*9 + 9)

    ASSERT_EQ(mlp.layers[0].in_dimension, 10)
    ASSERT_EQ(mlp.layers[0].out_dimension, 9)
    ASSERT_EQ(mlp.layers[0].activation_func, SIGMOID)
    ASSERT_EQ(mlp.layers[0].params.bias.size, 9)
    ASSERT_EQ(mlp.layers[0].params.weights.height, 9)
    ASSERT_EQ(mlp.layers[0].params.weights.width, 10)

    // Test with 2 layers
    mlp_builder_add_layer(&builder, 8, SOFTMAX, 0.2);
    mlp = mlp_builder_finalize(&builder);

    ASSERT_EQ(mlp.num_layers, 2)
    ASSERT_EQ(mlp.num_parameters, 10*9 + 9 + 9*8 + 8)

    ASSERT_EQ(mlp.layers[0].in_dimension, 10)
    ASSERT_EQ(mlp.layers[0].out_dimension, 9)
    ASSERT_EQ(mlp.layers[0].activation_func, SIGMOID)
    ASSERT_EQ(mlp.layers[0].params.bias.size, 9)
    ASSERT_EQ(mlp.layers[0].params.weights.height, 9)
    ASSERT_EQ(mlp.layers[0].params.weights.width, 10)

    ASSERT_EQ(mlp.layers[1].in_dimension, 9)
    ASSERT_EQ(mlp.layers[1].out_dimension, 8)
    ASSERT_EQ(mlp.layers[1].activation_func, SOFTMAX)
    ASSERT_EQ(mlp.layers[1].params.bias.size, 8)
    ASSERT_EQ(mlp.layers[1].params.weights.height, 8)
    ASSERT_EQ(mlp.layers[1].params.weights.width, 9)

    // Test with 3 layers
    mlp_builder_add_layer(&builder, 20, SIGMOID, 0.3);
    mlp = mlp_builder_finalize(&builder);

    ASSERT_EQ(mlp.num_layers, 3)
    ASSERT_EQ(mlp.num_parameters, 10*9 + 9 + 9*8 + 8 + 8*20 + 20)

    ASSERT_EQ(mlp.layers[0].in_dimension, 10)
    ASSERT_EQ(mlp.layers[0].out_dimension, 9)
    ASSERT_EQ(mlp.layers[0].activation_func, SIGMOID)
    ASSERT_EQ(mlp.layers[0].params.bias.size, 9)
    ASSERT_EQ(mlp.layers[0].params.weights.height, 9)
    ASSERT_EQ(mlp.layers[0].params.weights.width, 10)

    ASSERT_EQ(mlp.layers[1].in_dimension, 9)
    ASSERT_EQ(mlp.layers[1].out_dimension, 8)
    ASSERT_EQ(mlp.layers[1].activation_func, SOFTMAX)
    ASSERT_EQ(mlp.layers[1].params.bias.size, 8)
    ASSERT_EQ(mlp.layers[1].params.weights.height, 8)
    ASSERT_EQ(mlp.layers[1].params.weights.width, 9)

    ASSERT_EQ(mlp.layers[2].in_dimension, 8)
    ASSERT_EQ(mlp.layers[2].out_dimension, 20)
    ASSERT_EQ(mlp.layers[2].activation_func, SIGMOID)
    ASSERT_EQ(mlp.layers[2].params.bias.size, 20)
    ASSERT_EQ(mlp.layers[2].params.weights.height, 20)
    ASSERT_EQ(mlp.layers[2].params.weights.width, 8)
}


int main() {
    printf("test_create_mlp_builder\n");
    test_create_mlp_builder();

    printf("test_finalize_mlp_builder\n");
    test_finalize_mlp_builder();

    return 0;
}

