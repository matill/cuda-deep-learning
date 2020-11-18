#ifndef COST_FUNC_H
#define COST_FUNC_H

#include "linalg.h"


typedef enum {
    CROSS_ENTROPY   
} cost_func_t;


__global__ void cross_entropy_derivative(vector_t estimate, vector_t truth, vector_t estimate_gradient);

#endif
