#ifndef LINALG_H
#define LINALG_H

#include "common.h"


typedef struct device_matrix {
    u32 width;
    u32 height;
    u32 stride;
    f32 *vals;
} device_matrix_t;


typedef struct host_matrix {
    u32 width;
    u32 height;
    u32 stride;
    f32 *vals;
} host_matrix_t;


typedef struct device_vector {
    u32 size;
    f32 *vals;
} device_vector_t;


typedef struct host_vector {
    u32 size;
    f32 *vals;
} host_vector_t;


// Constructors
void vector_init(device_vector_t *vector, u32 size);
void matrix_init(device_matrix_t *matrix, u32 height, u32 width);
void vector_init_from_buf(device_vector_t *vector, u32 size, f32 **buf);
void matrix_init_from_buf(device_matrix_t *matrix, u32 height, u32 width, f32 **buf);
host_vector_t host_vector_init_static(u32 size, f32 *content);
host_vector_t host_vector_alloc(u32 size);
host_matrix_t host_matrix_init_static(u32 height, u32 width, f32 *buf);
host_matrix_t host_matrix_alloc(u32 height, u32 width);


// Data initializers
void vector_set_rand_unif_vals(device_vector_t *vector, f32 unif_low, f32 unif_high);
void matrix_set_rand_unif_vals(device_matrix_t *matrix, f32 unif_low, f32 unif_high);


// Copy functions
void vector_device_to_host(host_vector_t *host_vector, device_vector_t *device_vector);
void vector_host_to_device(device_vector_t *device_vector, host_vector_t *host_vector);
void matrix_device_to_host(host_matrix_t *host_matrix, device_matrix_t *device_matrix);
void matrix_host_to_device(device_matrix_t *device_matrix, host_matrix_t *host_matrix);


// Assertion functions
void assert_host_vectors_equal(host_vector_t a, host_vector_t b, char *file, u32 line);
void assert_host_matrices_equal(host_matrix_t a, host_matrix_t b, char *file, u32 line);
void assert_host_and_device_vectors_equal(host_vector_t a, device_vector_t b, char *file, u32 line);
void assert_host_and_device_matrices_equal(host_matrix_t a, device_matrix_t b, char *file, u32 line);
#define ASSERT_DEVICE_VECTOR(vector, s, ...) \
    { \
        f32 content_arr[s] = {__VA_ARGS__}; \
        host_vector_t expected = host_vector_init_static((s), content_arr); \
        assert_host_and_device_vectors_equal(expected, (vector), (char *) __FILE__, __LINE__); \
    }

#define ASSERT_DEVICE_MATRIX(matrix, height, width, ...) \
    { \
        f32 content_arr[(height) * (width)] = {__VA_ARGS__}; \
        host_matrix_t expected = host_matrix_init_static((height), (width), content_arr); \
        assert_host_and_device_matrices_equal(expected, (matrix), (char *) __FILE__, __LINE__); \
    }

__device__ inline f32 *matrix_index(device_matrix_t matrix, u32 i, u32 j) {
    return &matrix.vals[j + matrix.stride * i];
}

inline f32 *host_matrix_index(host_matrix_t matrix, u32 i, u32 j) {
    return &matrix.vals[j + matrix.stride * i];
}

__device__ void matrix_vector_multiply(device_matrix_t in_matrix, device_vector_t in_vector, device_vector_t out_vector);
__device__ f32 vector_dot(device_vector_t a, device_vector_t b);

__device__ void print_matrix(device_matrix_t matrix, char *name);
__device__ void print_vector(device_vector_t vector, char *name);
void host_print_matrix(device_matrix_t matrix, char *name);
void host_print_vector(device_vector_t vector, char *name);

#endif
