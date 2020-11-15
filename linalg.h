#ifndef LINALG_H
#define LINALG_H

#include "common.h"


typedef struct matrix {
    u32 width;
    u32 height;
    u32 stride;
    f32 *vals;
} matrix_t;


typedef struct vector {
    u32 size;
    f32 *vals;
} vector_t;


// Constructors
void vector_init(vector_t *vector, u32 size);
void matrix_init(matrix_t *matrix, u32 height, u32 width);

// Copy functions
void vector_device_to_host(vector_t *host_vector, vector_t *device_vector);
void vector_host_to_device(vector_t *device_vector, vector_t *host_vector);
void matrix_device_to_host(matrix_t *host_matrix, matrix_t *device_matrix);
void matrix_host_to_device(matrix_t *device_matrix, matrix_t *host_matrix);


// __device__ inline f32 *matrix_index(matrix_t matrix, u32 i, u32 j);
__device__ void matrix_vector_multiply(matrix_t in_matrix, vector_t in_vector, vector_t out_vector);

#endif
