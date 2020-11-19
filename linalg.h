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

// Copy functions
void vector_device_to_host(host_vector_t *host_vector, device_vector_t *device_vector);
void vector_host_to_device(device_vector_t *device_vector, host_vector_t *host_vector);
void matrix_device_to_host(host_matrix_t *host_matrix, device_matrix_t *device_matrix);
void matrix_host_to_device(device_matrix_t *device_matrix, host_matrix_t *host_matrix);


__device__ inline f32 *matrix_index(device_matrix_t matrix, u32 i, u32 j) {
    return &matrix.vals[i + matrix.stride * j];
}

__device__ void matrix_vector_multiply(device_matrix_t in_matrix, device_vector_t in_vector, device_vector_t out_vector);
__device__ f32 vector_dot(device_vector_t a, device_vector_t b);


#endif
