#include <stdio.h>

__global__
void kernel(int *arr)
{
    printf("threadIdx.x: %d\n", threadIdx.x);
    arr[threadIdx.x] = threadIdx.x;
}

int main(void)
{
    int *cuda_arr;
    cudaError_t error;
    error = cudaMalloc(&cuda_arr, sizeof(int) * 10);
    printf("malloc error: %d\n", error);
    kernel<<<10, 1>>>(cuda_arr);

    int host_arr[10];
        error = cudaMemcpy(host_arr, cuda_arr, 10*sizeof(int), cudaMemcpyDeviceToHost);
    printf("copy error: %d\n", error);
    for (int i = 0; i != 10; i++) {
        printf("arr[%d] = %d\n", i, host_arr[i]);
    }


//     int N = 1<<20;
//   float *x, *y, *d_x, *d_y;
//   x = (float*)malloc(N*sizeof(float));
//   y = (float*)malloc(N*sizeof(float));

//   cudaMalloc(&d_x, N*sizeof(float)); 
//   cudaMalloc(&d_y, N*sizeof(float));

//   for (int i = 0; i < N; i++) {
//     x[i] = 1.0f;
//     y[i] = 2.0f;
//   }

//   cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

//   // Perform SAXPY on 1M elements
//   saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

//   cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

//   float maxError = 0.0f;
//   for (int i = 0; i < N; i++)
//     maxError = max(maxError, abs(y[i]-4.0f));
//   printf("Max error: %f\n", maxError);

//   cudaFree(d_x);
//   cudaFree(d_y);
//   free(x);
//   free(y);
}