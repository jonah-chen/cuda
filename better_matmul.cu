#include <iostream>
#include <cstdlib>
#include <time.h>

#define TILE_WIDTH 32

/* multiplies matrices represented by a (m x n) and b (n x k) and assigns the value of ab to ans (m x k) matrix
 */
__global__
void matmul_old(float* a, float* b, float* ans, const int m, const int n, const int k)
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
void matmul(float* a, float* b, float* ab, int m, int n, int k)
{
    int row = threadIdx.x + blockIdx.x + blockDim.x;
    int col = threadIdx.y + blockIdx.x + blockDim.x;

    // a[row][col]= a[row*m+col], row<n, col<m
    // b[row][col]= b[row*n+col], row<k, col<n
    // ab[row][col]= ab[row*m+col], row<k, col<m

    if (row<k && col<m)
    {
        __shared__ float s_a[TILE_WIDTH][TILE_WIDTH]; 
        __shared__ float s_b[TILE_WIDTH][TILE_WIDTH]; 
        float result = 0.f;

        for (int p = 0; p < n/TILE_WIDTH; ++p)
        {
            s_a[threadIdx.x][threadIdx.y] = a[row*m + p*TILE_WIDTH+threadIdx.y];
            s_b[threadIdx.x][threadIdx.y] = b[n*(p*TILE_WIDTH+threadIdx.x) + col];
            __syncthreads();
            
            for (int i = 0; i < TILE_WIDTH; ++i)
                result += s_a[threadIdx.x][i] * s_b[i][threadIdx.y];
            __syncthreads();
        }
        ab[row*m+col] = result;              
    }
}

void matmul1(float* h_a, float* h_b, float* h_ans, const int m, const int n, const int k)
{
    float *d_a, *d_b, *d_ans;

    cudaMalloc((void**)&d_a, m*n*sizeof(float));
    cudaMalloc((void**)&d_b, n*k*sizeof(float));
    cudaMalloc((void**)&d_ans, m*k*sizeof(float));

    if (d_a == 0 || d_b == 0 || d_ans == 0)
    {
        printf("Failed to allocate device memory.\n");
        return;
    }

    // set the answer to all 0s
    cudaMemset(d_ans, 0, m*k*sizeof(float));

    // copy a and b to device
    cudaMemcpy(d_a, h_a, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*k*sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel
    matmul_old <<<(1023+m*k)/1024, 1024>>> (d_a, d_b, d_ans, m, n, k);

    // copy answer back to host
    cudaMemcpy(h_ans, d_ans, m*k*sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ans);
}

void matmul2(float* h_a, float* h_b, float* h_ans, const int m, const int n, const int k)
{
    float *d_a, *d_b, *d_ans;

    cudaMalloc((void**)&d_a, m*n*sizeof(float));
    cudaMalloc((void**)&d_b, n*k*sizeof(float));
    cudaMalloc((void**)&d_ans, m*k*sizeof(float));

    if (d_a == 0 || d_b == 0 || d_ans == 0)
    {
        printf("Failed to allocate device memory.\n");
        return;
    }

    // set the answer to all 0s
    cudaMemset(d_ans, 0, m*k*sizeof(float));

    // copy a and b to device
    cudaMemcpy(d_a, h_a, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*k*sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel
    dim3 block((31+m)/32,(31+k)/32,1);
    dim3 grid(32,32,1);

    matmul <<<block, grid, 2*TILE_WIDTH*TILE_WIDTH*sizeof(float)>>> (d_a, d_b, d_ans, m, n, k);

    // copy answer back to host
    cudaMemcpy(h_ans, d_ans, m*k*sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ans);
}

__global__
void add(float* a, float* ans)
{
    int b = blockIdx.x * blockDim.x;

    __shared__ float partial_ans[1024];
    {
        a[b+threadIdx.x]
    }
}


int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_ans1, *h_ans2;
    
    const int m = 1 << 9;
    const int n = 1 << 9;
    const int k = 1 << 9;

    
    // allocate memory
    h_a = (float*)calloc(m*n, sizeof(float));
    h_b = (float*)calloc(n*k, sizeof(float));
    h_ans1 = (float*)calloc(m*k, sizeof(float));
    h_ans2 = (float*)calloc(m*k, sizeof(float));

    // set random matrix
    srand (time(NULL));
    int t;
    t = m*n;
    while (t--)
        h_a[t] = -50.0 + (float)rand()*100.0/(float)RAND_MAX;
    t = n*k;
    while (t--)
        h_b[t] = -50.0 + (float)rand()*100.0/(float)RAND_MAX;

    matmul1(h_a, h_b, h_ans1, m, n, k);
    matmul2(h_a, h_b, h_ans2, m, n, k);

    for (int i = 0; i < m*k; ++i)
    {
        if(h_ans1[i] != h_ans2[i])
        {
            printf("Answer at index %d does not match. Old:%.1f New:%.1f\n", i, h_ans1[i], h_ans2[i]);
            break;
        }
    }
    
    // free host memory
    free(h_a);
    free(h_b);
    free(h_ans1);
    free(h_ans2);

    return 0;  
}