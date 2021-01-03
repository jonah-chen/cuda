#include <iostream>
#include <cstdlib>
#include <time.h>

#define TILE_WIDTH 32

/* multiplies matrices represented by a (m x n) and b (n x k) and assigns the value of ab to ans (m x k) matrix
 */
__global__
void matmul(float* a, float* b, float* ans, const int m, const int n, const int k)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < m*k)
    {      
        for (int j = 0; j < n; ++j)
        {
            int row = i/k;
            int col = i%k;
            ans[i] += a[n*row+j] * b[col+k*j];
        }
    }
}

__global__
void better_matmul(float* a, float* b, float* ans, const int m, const int n, const int k)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < m && col < k)
    {
        float res = 0.f;
        __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
        __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

        for (int i = 0; i < n/TILE_WIDTH; ++i)
        {
            s_a[threadIdx.x][threadIdx.y] = a[n*row+(i*TILE_WIDTH + threadIdx.y)];
            s_b[threadIdx.x][threadIdx.y] = b[(k*TILE_WIDTH + threadIdx.x)*n+col];
            __syncthreads();

            for (int j = 0; j < TILE_WIDTH; ++j)
                res += s_a[threadIdx.x][j] * s_b[j][threadIdx.y];
        }
    }
    
}

int main(int argc, char* argv[])
{    
    float *h_a, *h_b, *h_ans, *better_ans;
    
    float *d_a, *d_b, *d_ans;
    
    const int m = 1 << 9;
    const int n = 1 << 9;
    const int k = 1 << 9;

    // allocate GPU memory
    if (cudaMalloc(&d_a, m*n*sizeof(float)) != cudaSuccess)
        exit(1);
    if (cudaMalloc(&d_b, n*k*sizeof(float)) != cudaSuccess)
        exit(1);
    if (cudaMalloc(&d_ans, m*k*sizeof(float)) != cudaSuccess)
        exit(1);
    
    if (cudaMemset(d_ans, 0.0f, m*k*sizeof(float)) != cudaSuccess)
        exit(2);
    
    // allocate memory
    h_a = (float*)calloc(m*n, sizeof(float));
    h_b = (float*)calloc(n*k, sizeof(float));
    h_ans = (float*)calloc(m*k, sizeof(float));
    better_ans = (float*)calloc(m*k, sizeof(float));

    // set random matrix

    srand (time(NULL));
    int t;
    t = m*n;
    while (t--)
    {
        h_a[t] = -50.0 + (float)rand()*100.0/(float)RAND_MAX;
    }
    t = n*k;
    while (t--)
        h_b[t] = -50.0 + (float)rand()*100.0/(float)RAND_MAX;

    if (cudaMemcpy(d_a, h_a, m*n*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        exit(5);
    if (cudaMemcpy(d_b, h_b, n*k*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        exit(5);

    matmul<<<(1023+m*k)/1024, 1024>>>(d_a, d_b, d_ans, m, n, k);
    if (cudaMemcpy(h_ans, d_ans, m*k*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        exit(6);
    if (cudaMemset(d_ans, 0.0f, m*k*sizeof(float)) != cudaSuccess)
        exit(10);
    
    dim3 block((31+m)/32,(31+k)/32,1);
    dim3 grid(32,32,1);

    better_matmul <<<block, grid, 2*1024*sizeof(float)>>>(d_a, d_b, d_ans, m, n, k);
    if (cudaMemcpy(better_ans, d_ans, m*k*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        exit(11);
    
    // free GPU memory

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ans);

    // check if the outputs are the same

    for (int i = 0; i < m*k; ++i)
    {
        if (h_ans[i]!=better_ans[i])
        {
            printf("Results does not match at index %d. Old code: %.1f New code: %.1f", i, h_ans[i], better_ans[i]);
            break;
        }
    }
    
    // prints the inputs and outputs
    if (argc > 1)
    {
        printf("m: %d, n: %d, k: %d\n", m, n, k);

        printf("A is\n");
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                printf("%.1f ", h_a[i*n+j]);
            }
            printf("\n");
        }

        printf("B is\n");
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                printf("%.1f ", h_b[i*k+j]);
            }
            printf("\n");
        }

        printf("AB is\n");
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                printf("%.1f ", h_ans[i*k+j]);
            }
            printf("\n");
        }
    }

    // free host memory

    free(h_a);
    free(h_b);
    free(h_ans);
    free(better_ans);

    return 0;    
}