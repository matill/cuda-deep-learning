#include "cost_func.h"
#include "linalg.h"


inline __device__ f32 cross_entropy_derivative(vector_t estimate, vector_t truth) {
    // dJ/dy(L)_k = y_k / y(L)_k
    u32 k = threadIdx.x;
    f32 truth_k = truth.vals[k];
    f32 estimate_k = estimate.vals[k];
    return truth_k / estimate_k;
}


__device__ f32 compute_output_layer_y_gradient(vector_t y_out, vector_t y_out_expected, cost_func_t cost_func) {
    ASSERT_EQ_INT(y_out.size, y_out_expected.size);
    ASSERT_EQ_INT(y_out.size, blockDim.x);

    switch (cost_func) {
        case CROSS_ENTROPY:
            return cross_entropy_derivative(y_out, y_out_expected);

        default:
            printf("ERROR: cost_func: %d\n", cost_func);
            return 0;
    }
}

