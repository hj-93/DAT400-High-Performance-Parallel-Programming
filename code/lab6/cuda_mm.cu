#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define MATRIX_SIZE 1000
#define GRID_X 200
#define GRID_Y 200
#define GRID_Z 1
#define BLOCK_X 5
#define BLOCK_Y 5
#define BLOCK_Z 1
void printDeviceProp(const cudaDeviceProp &prop)
{
printf("Device Name : %s.\n", prop.name);
printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
printf("regsPerBlock : %d.\n", prop.regsPerBlock);
printf("warpSize : %d.\n", prop.warpSize);
printf("memPitch : %d.\n", prop.memPitch);
printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
printf("totalConstMem : %d.\n", prop.totalConstMem);
printf("major.minor : %d.%d.\n", prop.major, prop.minor);
printf("clockRate : %d.\n", prop.clockRate);
printf("textureAlignment : %d.\n", prop.textureAlignment);
printf("deviceOverlap : %d.\n", prop.deviceOverlap);
printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA Initialization
bool InitCUDA()
{
    int count;

    cudaGetDeviceCount(&count);

    if (count == 0) 
    {
        fprintf(stderr, "There is no device.\n");

        return false;
    }

    int i;

    for (i = 0; i < count; i++) 
    {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printDeviceProp(prop);

        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
        {
            if (prop.major >= 1) 
            {
            break;
            }
        }
    }

    if (i == count) 
    {
    fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
    return false;
    }

    cudaSetDevice(i);

    return true;

}

// Generate Random Matrix Elements
void matgen(float* a, int n)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {

            a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);

        }
    }
}

/* Task: Implement Your Kernel Function Here */
__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n)
{
    int g_threadId_x = blockIdx.x * blockDim.x + threadIdx.x;
    int g_threadId_y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    __shared__ float row_vec_shared[BLOCK_X * BLOCK_Y];
    __shared__ float col_vec_shared[BLOCK_X * BLOCK_Y];

    for (int i = 0; i < n/blockDim.x; i++) {
         row_vec_shared[threadIdx.x * blockDim.y + threadIdx.y] = a[g_threadId_x * n + i * blockDim.y + threadIdx.y];
         col_vec_shared[threadIdx.x * blockDim.y + threadIdx.y] = b[(i * blockDim.x + threadIdx.x) * n + g_threadId_y];
         __syncthreads();
         for (int j = 0; j < blockDim.x; j++) {
             c[g_threadId_x * n + g_threadId_y] += row_vec_shared[threadIdx.x * blockDim.y + j] * col_vec_shared[j* blockDim.y + threadIdx.y];
         }
        __syncthreads();
    }
}

int main()
{
    if (!InitCUDA()) return 0; 

    float *a, *b, *c, *d;

    int n = MATRIX_SIZE;

    a = (float*)malloc(sizeof(float)* n * n); 
    b = (float*)malloc(sizeof(float)* n * n); 
    c = (float*)malloc(sizeof(float)* n * n);
    d = (float*)malloc(sizeof(float)* n * n);
    srand(0);

    matgen(a, n);
    matgen(b, n);

    float *cuda_a, *cuda_b, *cuda_c;
    size_t size = sizeof(float) * n * n;
    /* Task: Memory Allocation */
    cudaMalloc((void **) &cuda_a, size);
    cudaMalloc((void **) &cuda_b, size);
    cudaMalloc((void **) &cuda_c, size);
    cudaMemset(&cuda_c, 0, size);
    /* Task: CUDA Memory Copy from Host to Device */
    cudaMemcpy(cuda_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, size, cudaMemcpyHostToDevice);

    /* Task: Number of Blocks and Threads && Dimention*/
    dim3 dimGrid(GRID_X, GRID_Y, GRID_Z);
    dim3 dimBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);

    // Kernel Execution
    matMultCUDA << < dimGrid, dimBlock >> >(cuda_a , cuda_b , cuda_c , n);

    /* Task: CUDA Memory Copy from Device to Host */
    cudaMemcpy(c, cuda_c, size, cudaMemcpyDeviceToHost);
    
    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    // CPU Implementation of MatMul
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        { 
            double t = 0;

            for (int k = 0; k < n; k++)
            { 

                t += a[i * n + k] * b[k * n + j]; 

            } 

            d[i * n + j] = t; 

        } 
    }

    // Check the accuracy of GPU results with CPU results
    float max_err = 0;
    float average_err = 0; 

    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            if (d[i * n + j] != 0)
            { 
                float err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);
                if (max_err < err) max_err = err; 
                average_err += err; 
            } 
        } 
    }

    printf("Max error: %g Average error: %g\n",max_err, average_err / (n * n));
return 0;
}
