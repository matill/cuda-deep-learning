#include "linalg.h"
#include "common.h"
#include "stdio.h"


void vector_init(vector_t *vector, u32 size) {
    cudaMalloc(&vector->vals, sizeof(f32) * size);
    vector->size = size;
}


void matrix_init(matrix_t *matrix, u32 height, u32 width) {
    cudaMalloc(&matrix->vals, sizeof(f32) * height * width);
    matrix->height = height;
    matrix->width = width;
    matrix->stride = width;
}


void vector_device_to_host(vector_t *host_vector, vector_t *device_vector) {
    ASSERT_EQ_INT(host_vector->size, device_vector->size);
    int x = cudaMemcpy(host_vector->vals, device_vector->vals, sizeof(f32) * host_vector->size, cudaMemcpyDeviceToHost);
    ASSERT_EQ_INT(x, 0);
}

void vector_host_to_device(vector_t *device_vector, vector_t *host_vector) {
    ASSERT_EQ_INT(host_vector->size, device_vector->size);
    int x = cudaMemcpy(device_vector->vals, host_vector->vals, sizeof(f32) * host_vector->size, cudaMemcpyHostToDevice);
    ASSERT_EQ_INT(x, 0);
}

static u64 matrix_copy_check(matrix_t *a, matrix_t *b) {
    ASSERT_EQ_INT(a->height, b->height);
    ASSERT_EQ_INT(a->stride, b->stride);
    ASSERT_EQ_INT(a->width, b->width);
    return sizeof(f32) * a->height * a->stride;
}

void matrix_device_to_host(matrix_t *host_matrix, matrix_t *device_matrix) {
    u64 num_bytes = matrix_copy_check(host_matrix, device_matrix);
    int x = cudaMemcpy(host_matrix->vals, device_matrix->vals, num_bytes, cudaMemcpyDeviceToHost);
    ASSERT_EQ_INT(x, 0);
}

void matrix_host_to_device(matrix_t *device_matrix, matrix_t *host_matrix) {
    u64 num_bytes = matrix_copy_check(host_matrix, device_matrix);
    int x = cudaMemcpy(device_matrix->vals, host_matrix->vals, num_bytes, cudaMemcpyHostToDevice);
    ASSERT_EQ_INT(x, 0);
}


__device__ inline f32 *matrix_index(matrix_t matrix, u32 i, u32 j) {
    return &matrix.vals[i + matrix.stride * j];
}

__device__ void matrix_vector_multiply(matrix_t in_matrix, vector_t in_vector, vector_t out_vector) {
    ASSERT_EQ_INT(in_matrix.width, in_vector.size);
    ASSERT_EQ_INT(in_matrix.height, out_vector.size);
    ASSERT_EQ_INT(out_vector.size, blockDim.x);
    ASSERT_EQ_INT(1, blockDim.y);
    ASSERT_EQ_INT(1, blockDim.z);

    // Compute value
    f32 thread_val = 0;
    for (u32 i = 0; i != in_vector.size; i++) {
        f32 matrix_val = *matrix_index(in_matrix, threadIdx.x, i);
        f32 vector_val = in_vector.vals[i];
        thread_val += matrix_val * vector_val;
    }
    out_vector.vals[threadIdx.x] = thread_val;
}

