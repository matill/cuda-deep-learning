#ifndef ACTIVATION_FUNC_H
#define ACTIVATION_FUNC_H


#include "common.h"
#include "linalg.h"


typedef enum {
    SIGMOID, SOFTMAX
} activation_func_t;


// Used for forward computation.
__device__ void apply_activation_func(device_vector_t vector, activation_func_t activation_func);


// Used at back propagation.
// y_gradient: vector that can be used to store all y_k_derivative values since softmax needs all values and not just the one
// corresponding to the same index. The buffer is uninitialized, since it is in device-global memory, and is therefore slower.
__device__ f32 transform_yk_derivative_to_vk_derivative(device_vector_t y, f32 y_k_derivative, device_vector_t y_gradient, activation_func_t activation_func);


#endif
