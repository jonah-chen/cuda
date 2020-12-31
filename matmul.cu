#include <iostream>
#include <cstdlib>
#include <time.h>

/* multiplies matrices represented by a (m x n) and b (n x k) and assigns the value of ab to ans (m x k) matrix
 */
__global__
void matmul(float* a, float* b, float *ans, const int m, const int n, const int k)
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

int main(int argc, char* argv[])
{    
    float *h_a, *h_b, *h_ans;
    
    float *d_a, *d_b, *d_ans;
    
    const int m = 1 << 9;
    const int n = 1 << 9;
    const int k = 1 << 9;

    // allocate GPU memory
    if (cudaMalloc(&d_a, m*n*sizeof(float)) != cudaSuccess)
        return -1;
    if (cudaMalloc(&d_b, n*k*sizeof(float)) != cudaSuccess)
        return -1;
    if (cudaMalloc(&d_ans, m*k*sizeof(float)) != cudaSuccess)
        return -1;
    
    if (cudaMemset(d_ans, 0.0f, m*k*sizeof(float)) != cudaSuccess)
        return -2;
    
    // allocate memory
    h_a = (float*)calloc(m*n, sizeof(float));
    h_b = (float*)calloc(n*k, sizeof(float));
    h_ans = (float*)calloc(m*k, sizeof(float));

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
        return -4;
    if (cudaMemcpy(d_b, h_b, n*k*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return -4;

    matmul<<<(1023+m*k)/1024, 1024>>>(d_a, d_b, d_ans, m, n, k);

    // copy stuff

    if (cudaMemcpy(h_ans, d_ans, m*k*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -3;
    
    // free GPU memory

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ans);
    
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

    return 0;    
}