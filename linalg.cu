#include "linalg.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
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


host_vector_t host_vector_init_static(u32 size, f32 *content) {
    host_vector_t vector = {
        .size = size,
        .vals = content
    };
    return vector;
}


host_vector_t host_vector_alloc(u32 size) {
    host_vector_t vector = {
        .size = size,
        .vals = (f32 *) malloc(sizeof(f32) * size)
    };
    ASSERT_NOT_NULL(vector.vals);
    return vector;
}


host_matrix_t host_matrix_init_static(u32 height, u32 width, f32 *buf) {
    host_matrix_t matrix = {
        .width = width,
        .height = height,
        .stride = width,
        .vals = buf
    };
    return matrix;
}


host_matrix_t host_matrix_alloc(u32 height, u32 width) {
    host_matrix_t matrix = {
        .width = width,
        .height = height,
        .stride = width,
        .vals = (f32 *) malloc(sizeof(f32) * height * width)
    };
    ASSERT_NOT_NULL(matrix.vals)
    return matrix;
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


u32 float_eq(f32 a, f32 b) {
    const f32 accuracy = 0.0001;
    f32 diff = a - b;
    return -accuracy < diff && diff < accuracy;
} 


// Assertion functions
void assert_host_vectors_equal(host_vector_t a, host_vector_t b, char *file, u32 line) {
    if (a.size != b.size) {
        printf("Assertion error %s:%d. Incompatible shapes. %d != %d\n", file, line, a.size, b.size);
        exit(-1);
    }

    u32 num_errors = 0;
    for (u32 i = 0; i != a.size; i++) {
        if (!float_eq(a.vals[i], b.vals[i])) {
            printf("Assertion error %s:%d. In element %d: %f != %f\n", file, line, i, a.vals[i], b.vals[i]);
            num_errors++;
        }
    }

    if (num_errors > 0) {
        exit(-1);
    }
}


void assert_host_matrices_equal(host_matrix_t a, host_matrix_t b, char *file, u32 line) {
    if (a.height != b.height || a.width != b.width) {
        printf("Assertion error %s:%d. Incompatible shapes. (%d, %d) != (%d, %d)\n", file, line, a.height, a.width, b.height, b.width);
        exit(-1);
    }

    u32 num_errors = 0;
    for (u32 i = 0; i != a.height; i++) {
        for (u32 j = 0; j != a.width; j++) {
            f32 a_ij = *host_matrix_index(a, i, j);
            f32 b_ij = *host_matrix_index(b, i, j);
            if (!float_eq(a_ij, b_ij)) {
                printf("Assertion error %s:%d. In element (%d, %d): %f != %f\n", file, line, i, j, a_ij, b_ij);
                num_errors++;
            }
        }
    }

    if (num_errors > 0) {
        exit(-1);
    }
}


void assert_host_and_device_vectors_equal(host_vector_t a, device_vector_t b, char *file, u32 line) {
    host_vector_t b_host = host_vector_alloc(b.size);
    vector_device_to_host(&b_host, &b);
    assert_host_vectors_equal(a, b_host, file, line);
}


void assert_host_and_device_matrices_equal(host_matrix_t a, device_matrix_t b, char *file, u32 line) {
    host_matrix_t b_host = host_matrix_alloc(b.height, b.width);
    matrix_device_to_host(&b_host, &b);
    assert_host_matrices_equal(a, b_host, file, line);
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


__device__ void print_matrix_core(device_matrix_t matrix) {
    for (u32 i = 0; i != matrix.height; i++) {
        printf("[");
        for (u32 j = 0; j != matrix.width; j++) {
            printf("%f", *matrix_index(matrix, i, j));
            if (j == matrix.width - 1) {
                printf("]\n");
            } else {
                printf("\t");
            }
        }
    }
}


__device__ void print_vector_core(device_vector_t vector) {
    for (u32 i = 0; i != vector.size; i++) {
        printf("[%f]\n", vector.vals[i]);
    }
}


__device__ void print_matrix(device_matrix_t matrix, char *name) {
    if (threadIdx.x == 0) {
        printf("%s: (%d, %d):\n", name, matrix.height, matrix.width);
        print_matrix_core(matrix);
    }
}


__device__ void print_vector(device_vector_t vector, char *name) {
    if (threadIdx.x == 0) {
        printf("%s: %d:\n", name, vector.size);
        print_vector_core(vector);
    }
}


__global__ void host_print_matrix_kernel(device_matrix_t matrix) {
    print_matrix_core(matrix);
}

__global__ void host_print_vector_kernel(device_vector_t vector) {
    print_vector_core(vector);
}

void host_print_matrix(device_matrix_t matrix, char *name) {
    printf("%s: (%d, %d):\n", name, matrix.height, matrix.width);
    host_print_matrix_kernel<<<1, 1>>>(matrix);
    cudaDeviceSynchronize();
}


void host_print_vector(device_vector_t vector, char *name) {
    printf("%s: %d:\n", name, vector.size);
    host_print_vector_kernel<<<1, 1>>>(vector);
    cudaDeviceSynchronize();
}


