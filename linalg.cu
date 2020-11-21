#include "linalg.h"
#include "common.h"
#include "stdio.h"
#include <random>


// Constructors
void vector_init(device_vector_t *vector, u32 size) {
    cudaMalloc(&vector->vals, sizeof(f32) * size);
    vector->size = size;
}


void matrix_init(device_matrix_t *matrix, u32 height, u32 width) {
    cudaMalloc(&matrix->vals, sizeof(f32) * height * width);
    matrix->height = height;
    matrix->width = width;
    matrix->stride = width;
}


void vector_init_from_buf(device_vector_t *vector, u32 size, f32 **buf) {
    vector->vals = *buf;
    *buf += size;
    vector->size = size;
}


void matrix_init_from_buf(device_matrix_t *matrix, u32 height, u32 width, f32 **buf) {
    matrix->vals = *buf;
    *buf += height * width;
    matrix->height = height;
    matrix->width = width;
    matrix->stride = width;
}


// Data initializers
static f32 *generate_rand_unif_array(u32 num_vals, f32 unif_low, f32 unif_high) {
    std::default_random_engine generator;
    std::uniform_real_distribution<f32> distribution(unif_low, unif_high);
    f32 *data = (f32 *) malloc(sizeof(f32) * num_vals);
    ASSERT_NOT_NULL(data);
    for (u32 i = 0; i != num_vals; i++) {
        data[i] = distribution(generator);
    }

    return data;
}

void vector_set_rand_unif_vals(device_vector_t *vector, f32 unif_low, f32 unif_high) {
    f32 *data = generate_rand_unif_array(vector->size, unif_low, unif_high);
    i32 x = cudaMemcpy(vector->vals, data, vector->size * sizeof(f32), cudaMemcpyHostToDevice);
    free(data);
    ASSERT_EQ_INT(x, 0);
}


// NOTE: Only works with matrices with equal stride and width (which is most often the case)
void matrix_set_rand_unif_vals(device_matrix_t *matrix, f32 unif_low, f32 unif_high) {
    ASSERT_EQ_INT(matrix->width, matrix->stride);
    u32 size = matrix->height * matrix->width;
    f32 *data = generate_rand_unif_array(size, unif_low, unif_high);
    i32 x = cudaMemcpy(matrix->vals, data, size * sizeof(f32), cudaMemcpyHostToDevice);
    free(data);
    ASSERT_EQ_INT(x, 0);
}


void vector_device_to_host(host_vector_t *host_vector, device_vector_t *device_vector) {
    ASSERT_EQ_INT(host_vector->size, device_vector->size);
    int x = cudaMemcpy(host_vector->vals, device_vector->vals, sizeof(f32) * host_vector->size, cudaMemcpyDeviceToHost);
    ASSERT_EQ_INT(x, 0);
}

void vector_host_to_device(device_vector_t *device_vector, host_vector_t *host_vector) {
    ASSERT_EQ_INT(host_vector->size, device_vector->size);
    int x = cudaMemcpy(device_vector->vals, host_vector->vals, sizeof(f32) * host_vector->size, cudaMemcpyHostToDevice);
    ASSERT_EQ_INT(x, 0);
}

static u64 matrix_copy_check(host_matrix_t *host_matrix, device_matrix_t *device_matrix) {
    ASSERT_EQ_INT(host_matrix->height, device_matrix->height);
    ASSERT_EQ_INT(host_matrix->stride, device_matrix->stride);
    ASSERT_EQ_INT(host_matrix->width, device_matrix->width);
    return sizeof(f32) * host_matrix->height * host_matrix->stride;
}

void matrix_device_to_host(host_matrix_t *host_matrix, device_matrix_t *device_matrix) {
    u64 num_bytes = matrix_copy_check(host_matrix, device_matrix);
    int x = cudaMemcpy(host_matrix->vals, device_matrix->vals, num_bytes, cudaMemcpyDeviceToHost);
    ASSERT_EQ_INT(x, 0);
}

void matrix_host_to_device(device_matrix_t *device_matrix, host_matrix_t *host_matrix) {
    u64 num_bytes = matrix_copy_check(host_matrix, device_matrix);
    int x = cudaMemcpy(device_matrix->vals, host_matrix->vals, num_bytes, cudaMemcpyHostToDevice);
    ASSERT_EQ_INT(x, 0);
}

__device__ void matrix_vector_multiply(device_matrix_t in_matrix, device_vector_t in_vector, device_vector_t out_vector) {
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

// TODO: Optimize
__device__ f32 vector_dot(device_vector_t a, device_vector_t b) {
    ASSERT_EQ_INT(a.size, b.size);
    f32 sum = 0;
    for (u32 i = 0; i != a.size; i++) {
        sum += a.vals[i] * b.vals[i];
    }

    return sum;
}