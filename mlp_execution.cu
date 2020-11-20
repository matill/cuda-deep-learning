
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
