#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>

/* Generates a mandelbrot set image
 *
 * The elements of bounds is the real min/max and imaginary min/max in that order
 * iterations is the number of iterations of z^2+c on each pixel
 * The percision is the real and imaginary step size in that order
 */
__global__ 
void mandelbrot(unsigned int* image, 
    const double left, 
    const double right, 
    const double down, 
    const double up, 
    const int iterations, 
    const double real_precision, 
    const double im_precision)
{
    // Representation of the image is in row-major order. Thus, the (re,im) pixel is image[re+im*re_size]
    
    int x = threadIdx.x + blockIdx.x * blockDim.x; // real part
    int y = threadIdx.y + blockIdx.y * blockDim.y; // imaginary part

    const int re_size = (int)((right-left)/real_precision);
    const int im_size = (int)((up-down)/im_precision);

    if (x < re_size && y < im_size) // check that it is in bounds
    {
        // c=r+it is the original number
        double r = left + real_precision*x;
        double t = down + im_precision*y;

        // start from z=0
        double zr = 0.0;
        double zt = 0.0;
        // temporary variables
        double _zr, _zt;

        for (int iteration = 0; iteration < iterations-1; ++iteration)
        {
            // perform an iteration of z^2+c
            _zr = zr*zr - zt*zt + r;
            _zt = zr*zt*2 + t;
            zr = _zr;
            zt = _zt;
            // check for |z|>2 (or |z|^2>4)
            if (zr*zr+zt*zt > 4.0)
            {
                // convert to image. here, a linear gamma curve is used
                image[x+y*re_size] = iteration;
                return;
            }
        }
        image[x+y*re_size] = iterations-1;
    }
}

int check(double r, double t, int iterations)
{
    double zr = 0.0;
    double zt = 0.0;
    double _zr, _zt;
    for (int iteration = 0; iteration < iterations-1; ++iteration)
    {
        // perform an iteration of z^2+c
        _zr = zr*zr - zt*zt + r;
        _zt = zr*zt*2 + t;
        zr = _zr;
        zt = _zt;
        printf("%.2f+%.2fi\n", zr, zt);

        // check for |z|>2 (or |z|^2>4)
        if (zr*zr+zt*zt > 4.0)
        {
            // convert to image. here, a linear gamma curve is used
            return iteration;
        }
    }
    return iterations-1;
}

int main(int argc, char* argv[])
{
    // the default bounds for the mandelbrot set is usually -2<Re(z)<1 and -1<Im(z)<1
    double bounds[4] = {-2.0, 1.0, -1.0, 1.0};
    
    // given precision of 0.001, it will generate 3000x2000 image which is about 6 megapixel
    double precision[2] = {0.001, 0.001};
    
    // i feel like making iterate 256 times
    int iterations = 256;

    unsigned int *d_image, *h_image;
    int re_size, im_size;
    
    // compute the size of the image
    re_size = (int)((bounds[1]-bounds[0])/precision[0]);
    im_size = (int)((bounds[3]-bounds[2])/precision[1]);
    
    // allocate the GPU memory for the image
    if (cudaMalloc(&d_image, re_size*im_size*sizeof(int)) != cudaSuccess) 
    {
        std::cout << "The device does not have enough memory. Program exited with error code -1." << std::endl;
        return -1;
    }

    // set the memory to 0
    if (cudaMemset(d_image, 0, re_size*im_size*sizeof(int)) != cudaSuccess)
    {
        std::cout << "Failed to set memory to 0. Program exited with error code -2." << std::endl;
        return -2;
    }

    // create the grid and blocks to call the method. 32x32=1024 threads.
    dim3 grid((re_size+31)/32, (im_size+31)/32, 1);
    dim3 block(32,32,1);
    mandelbrot <<<grid, block>>> (d_image, bounds[0], bounds[1], bounds[2], bounds[3], iterations, precision[0], precision[1]);
    cudaDeviceSynchronize();

    h_image = (unsigned int*)calloc(re_size*im_size, sizeof(int));
    cudaMemcpy(h_image, d_image, re_size*im_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_image);

    std::cout << h_image[2000+1000*re_size] << std::endl;

    // check against CPU code
    srand(time(NULL));
    for (int i = 0; i < 1; ++i)
    {
        int x = rand() % re_size;
        int y = rand() % im_size;

        printf("z=%.2f+%.2fi GPU:%d CPU:%d\n", bounds[0]+(double)x*precision[0], bounds[2]+(double)y*precision[1], h_image[x+y*re_size], check(bounds[0]+(double)x*precision[0], bounds[2]+(double)y*precision[1], iterations));
    }

    // write the image
    std::ofstream img ("mandelbrot.ppm");
    img << "P3" << std::endl;
    img << re_size << " " << im_size << std::endl;
    img << iterations-1 << std::endl;
    for (int y = 0; y < im_size; ++y)
    {
        for (int x = 0; x < re_size; ++x)
        {
            img << h_image[x+y*re_size] << " " << h_image[x+y*re_size] << " " << h_image[x+y*re_size] << std::endl;
        }
    }


    free(h_image);


    return 0;
}