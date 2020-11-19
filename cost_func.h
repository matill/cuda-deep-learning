#ifndef COST_FUNC_H
#define COST_FUNC_H

#include "linalg.h"


typedef enum {
    CROSS_ENTROPY   
} cost_func_t;


__device__ f32 compute_output_layer_y_gradient(device_vector_t estimate, device_vector_t truth, cost_func_t cost_func);


#endif
