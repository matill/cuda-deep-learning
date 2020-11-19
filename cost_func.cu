#include "cost_func.h"
#include "linalg.h"


__global__ void cross_entropy_derivative(vector_t estimate, vector_t truth, vector_t estimate_gradient) {
    // dJ/dy(L)_k = y_k / y(L)_k
    ASSERT_EQ_INT(estimate.size, truth.size);
    ASSERT_EQ_INT(estimate.size, estimate_gradient.size);
    ASSERT_EQ_INT(estimate.size, blockDim.x);
    u32 k = threadIdx.x;
    f32 truth_k = truth.vals[k];
    f32 estimate_k = estimate.vals[k];
    estimate_gradient.vals[k] = truth_k / estimate_k;
}


__device__ f32 compute_output_layer_y_gradient(vector_t estimate, vector_t truth, cost_func_t cost_func) {

}

