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
void mandelbrot(unsigned char* image, 
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

        // start from z=0 and declare temporary variables
        double zr = 0.0, zt = 0.0, _zr, _zt;

        for (int iteration = 1; iteration < iterations; ++iteration)
        {
            // perform an iteration of z^2+c
            _zr = zr*zr - zt*zt + r;
            _zt = zr*zt*2 + t;
            zr = _zr;
            zt = _zt;
            // check for |z|>2 (or |z|^2>4)
            if (zr*zr+zt*zt > 4.0)
            {
                // convert to image using a log distribution.
                int q = (int)((sqrtf(iteration)*512)/sqrtf(iterations));
                int s = 3*(x+y*re_size); // s:red, s+1:green, s+2:blue
                if (q < 256)
                {
                    image[s+2] = 255-q;
                    image[s] = q;
                }
                else
                {
                    image[s] = 511-q;
                    image[s+1] = q-256;
                }                
                return;
            }
        }
        // If it never reaches |z|>2, the point is then in the mandobrot set given the limitations of the program.
        // Therefore, the point is represented in the image as black
    }
}

int check(double r, double t, int iterations)
{
    double zr = 0.0, zt = 0.0, _zr, _zt;
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
            return iteration;
        }
    }
    return iterations-1;
}

int main(int argc, char* argv[])
{
    // the default bounds for the mandelbrot set is usually -2<Re(z)<1 and -1<Im(z)<1
    double bounds[4] = {-0.235125-4e-5, -0.235125+12e-5, 0.827215-4e-5, 0.827215+7.5e-5};
    
    // given precision of 0.001, it will generate 3000x2000 image which is about 6 megapixel
    double precision[2] = {1e-8, 1e-8};
    
    // this just fits the 3 8-bit channels
    int iterations = 1 << 11;

    unsigned char *d_image, *h_image;
    int re_size, im_size;
    
    // compute the size of the image
    re_size = (int)((bounds[1]-bounds[0])/precision[0]);
    im_size = (int)((bounds[3]-bounds[2])/precision[1]);
    
    // allocate the GPU memory for the image
    if (cudaMalloc(&d_image, 3*re_size*im_size*sizeof(char)) != cudaSuccess) 
    {
        std::cout << "The device does not have enough memory. Program exited with error code -1." << std::endl;
        return -1;
    }

    // set the memory to 0
    if (cudaMemset(d_image, 0, 3*re_size*im_size*sizeof(char)) != cudaSuccess)
    {
        std::cout << "Failed to set memory to 0. Program exited with error code -2." << std::endl;
        return -2;
    }

    // create the grid and blocks to call the method. 32x32=1024 threads.
    dim3 grid((re_size+31)/32, (im_size+31)/32, 1);
    dim3 block(32,32,1);
    mandelbrot <<<grid, block>>> (d_image, bounds[0], bounds[1], bounds[2], bounds[3], iterations, precision[0], precision[1]);
    cudaDeviceSynchronize();

    h_image = (unsigned char*)calloc(3*re_size*im_size, sizeof(char));
    if (cudaMemcpy(h_image, d_image, 3*re_size*im_size*sizeof(char), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cout << "Failed to copy the memory from device to host. Program exited with error code -9";
        return -9;
    }
    cudaFree(d_image);

    #ifdef DEBUG
    // check against CPU code
    srand(time(NULL));
    for (int i = 0; i < 20; ++i)
    {
        int x = rand() % re_size;
        int y = rand() % im_size;

        printf("z=%.2f+%.2fi GPU:%d CPU:%d\n", bounds[0]+(double)x*precision[0], bounds[2]+(double)y*precision[1], h_image[3*(x+y*re_size)], check(bounds[0]+(double)x*precision[0], bounds[2]+(double)y*precision[1], iterations));
    }
    #endif

    // write the image
    FILE *img;
    if (argc > 1)
        img = fopen(argv[1], "wb");
    else
        img = fopen("/mnt/ramdisk/image.ppm", "wb");
    fprintf(img, "P6\n%d %d\n%d\n", re_size, im_size, 255);
    fwrite(h_image, sizeof(char), 3*re_size*im_size, img);
    fclose(img);
    
    free(h_image);

    return 0;
}