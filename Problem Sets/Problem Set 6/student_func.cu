//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"
#include <stdio.h>
#include "timer.h"

#define uchar unsigned char

#define MAX_THREADS 1024
#define SQRT_MAX_THREADS 32

#define convert2Dto1D(x, y, numCols) (x) + (y) * (numCols)

#define PROFILE 1

#if PROFILE
  #define STARTCLOCK(name) GpuTimer name; name.Start();
  #define ENDCLOCK(name) name.Stop(); printf("%s: %f\n", #name, name.Elapsed());
#else
  #define STARTCLOCK(name) 
  #define ENDCLOCK(name) 
#endif

__global__
void mask(char* mask, uchar4* source, int numPixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numPixels) {
    uchar4 pixel = source[idx];
    mask[idx] = pixel.x != 255 || pixel.y != 255 || pixel.z != 255;
  }
}

__global__
void findBorderAndInterior(char* border, char* interior, char* mask, size_t numCols, size_t numRows) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < numCols && row < numRows) {
    int idx = col + row * numCols;
    int inMask = mask[idx];
    int numNeighborsInMask = 0;

    idx = (col - 1) + row * numCols;
    if (mask[idx]) {
      numNeighborsInMask++;
    }

    idx = (col + 1) + row * numCols;
    if (mask[idx]) {
      numNeighborsInMask++;
    }

    idx = col + (row - 1) * numCols;
    if (mask[idx]) {
      numNeighborsInMask++;
    }

    idx = col + (row + 1) * numCols;
    if (mask[idx]) {
      numNeighborsInMask++;
    }

    idx = col + row * numCols;
    interior[idx] = inMask && (numNeighborsInMask == 4);
    border[idx] = inMask && (numNeighborsInMask >= 1) && (numNeighborsInMask < 4);
  }
}

__global__
void checkMask(uchar4* destImg, char* mask, uchar4* blendedImg, int numPixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numPixels) {
    uchar4 newVal;

    if (mask[idx]) {
      newVal = destImg[idx];
    }
    else {
      newVal.x = newVal.y = newVal.z = 0;
    }

    blendedImg[idx] = newVal;
  }
}

__global__
void separateChannels(uchar4* combined, uchar* r, uchar* g, uchar* b, int numPixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numPixels) {
    uchar4 pixel = combined[idx];

    r[idx] = pixel.x;
    g[idx] = pixel.y;
    b[idx] = pixel.z;
  }
}

__global__
void recombineChannels(uchar4* combined, uchar* r, uchar* g, uchar* b, int numPixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numPixels) {
    uchar4 pixel;

    pixel.x = r[idx];
    pixel.y = g[idx];
    pixel.z = b[idx];

    combined[idx] = pixel;
  }
}

__global__
void initBuffer(float* dest, uchar* src, int numPixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numPixels) {
    dest[idx] = (float)src[idx];
  }
}

__global__
void jacobi(float* imgGuessPrev, float* imgGuessNext, char* interior, unsigned char* sourceImg, unsigned char* dstImg, size_t numCols, size_t numRows) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

  int pIdx = col + row * numCols;

  if (col < numCols && row < numRows && interior[pIdx]) {
    float sum1 = 0;
    float sum2 = 0;

    uchar pImg = sourceImg[pIdx];
    
    int idx = (col - 1) + row * numCols;
    if (interior[idx]) {
      sum1 += imgGuessPrev[idx];
    }
    else {
      sum1 += dstImg[idx];
    }
    sum2 += pImg - sourceImg[idx];

    idx = (col + 1) + row * numCols;
    if (interior[idx]) {
      sum1 += imgGuessPrev[idx];
    }
    else {
      sum1 += dstImg[idx];
    }
    sum2 += pImg - sourceImg[idx];

    idx = col + (row - 1) * numCols;
    if (interior[idx]) {
      sum1 += imgGuessPrev[idx];
    }
    else {
      sum1 += dstImg[idx];
    }
    sum2 += pImg - sourceImg[idx];

    idx = col + (row + 1) * numCols;
    if (interior[idx]) {
      sum1 += imgGuessPrev[idx];
    }
    else {
      sum1 += dstImg[idx];
    }
    sum2 += pImg - sourceImg[idx];

    float newVal = (sum1 + sum2) / 4.0f;

    if (newVal > 255.0f) {
      newVal = 255.0f;
    }

    if (newVal < 0.0f) {
      newVal = 0.0f;
    }

    imgGuessNext[pIdx] = newVal;
  }
}

__global__
void finalizeJacobi(float* imgGuessPrev, uchar* srcImg, uchar* dstImg, char* interior, int numPixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numPixels) {
    if (interior[idx]) {
      dstImg[idx] = (uchar)imgGuessPrev[idx];
    }
  }
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
  // step zero - book keeping
  STARTCLOCK(step_zero);

  const int numPixels = numRowsSource * numColsSource;
  const size_t imageSize = numPixels * sizeof(uchar4);

  uchar4 *d_sourceImg;  // get h_sourceImg, h_destImg, and h_blendedImg on device
  uchar4 *d_destImg;
  uchar4 *d_blendedImg;

  checkCudaErrors(cudaMalloc(&d_sourceImg, imageSize));
  checkCudaErrors(cudaMalloc(&d_destImg, imageSize));
  checkCudaErrors(cudaMalloc(&d_blendedImg, imageSize));

  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, imageSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, imageSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_blendedImg, h_destImg, imageSize, cudaMemcpyHostToDevice));

  ENDCLOCK(step_zero);

  // step one - compute mask for source image
  STARTCLOCK(step_one);

  char* d_sourceMask;   // alocate mask on device
  checkCudaErrors(cudaMalloc(&d_sourceMask, sizeof(char) * numPixels));

  const size_t threads = MAX_THREADS;
  const size_t blocks = (float)numPixels / MAX_THREADS + 0.5;

  mask<<<blocks, threads>>>(d_sourceMask, d_sourceImg, numPixels);  // fill in mask
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //checkCudaErrors(cudaMemcpy(d_blendedImg, d_destImg, imageSize, cudaMemcpyDeviceToDevice));  // test passed
  //checkMask<<<blocks, threads>>>(d_destImg, d_sourceMask, d_blendedImg, numPixels);           // mask computed correctly

  ENDCLOCK(step_one);

  // step two - compute interior and border regions
  STARTCLOCK(step_two);

  char* d_border;   // alocated arrays for border and interior
  char* d_interior;

  checkCudaErrors(cudaMalloc(&d_border, sizeof(char) * numPixels));
  checkCudaErrors(cudaMalloc(&d_interior, sizeof(char) * numPixels));

  dim3 step2Threads(SQRT_MAX_THREADS, SQRT_MAX_THREADS);
  dim3 step2Blocks(numColsSource / step2Threads.x + 1, numRowsSource / step2Threads.y + 1);

  findBorderAndInterior<<<step2Blocks, step2Threads>>>(d_border, d_interior, d_sourceMask, numColsSource, numRowsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //checkMask<<<blocks, threads>>>(d_destImg, d_interior, d_blendedImg, numPixels); // after eyeballing it looked correct

  ENDCLOCK(step_two);

  // step three - separate channels
  STARTCLOCK(step_three);

  unsigned char* d_sourceImg_r;
  unsigned char* d_sourceImg_g;
  unsigned char* d_sourceImg_b;

  unsigned char* d_dstImg_r;
  unsigned char* d_dstImg_g;
  unsigned char* d_dstImg_b;

  size_t channelSize = sizeof(unsigned char) * numPixels;

  checkCudaErrors(cudaMalloc(&d_sourceImg_r, channelSize));
  checkCudaErrors(cudaMalloc(&d_sourceImg_g, channelSize));
  checkCudaErrors(cudaMalloc(&d_sourceImg_b, channelSize));

  checkCudaErrors(cudaMalloc(&d_dstImg_r, channelSize));
  checkCudaErrors(cudaMalloc(&d_dstImg_g, channelSize));
  checkCudaErrors(cudaMalloc(&d_dstImg_b, channelSize));

  separateChannels<<<blocks, threads>>>(d_sourceImg, d_sourceImg_r, d_sourceImg_g, d_sourceImg_b, numPixels);
  separateChannels<<<blocks, threads>>>(d_destImg, d_dstImg_r, d_dstImg_g, d_dstImg_b, numPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  ENDCLOCK(step_three);

  // step four - initialize float buffers
  STARTCLOCK(step_four);

  size_t bufferSize = sizeof(float) * numPixels;

  float* d_imageGuessPrev_r;
  float* d_imageGuessPrev_g;
  float* d_imageGuessPrev_b;

  checkCudaErrors(cudaMalloc(&d_imageGuessPrev_r, bufferSize));
  checkCudaErrors(cudaMalloc(&d_imageGuessPrev_g, bufferSize));
  checkCudaErrors(cudaMalloc(&d_imageGuessPrev_b, bufferSize));

  float* d_imageGuessNext_r;
  float* d_imageGuessNext_g;
  float* d_imageGuessNext_b;

  checkCudaErrors(cudaMalloc(&d_imageGuessNext_r, bufferSize));
  checkCudaErrors(cudaMalloc(&d_imageGuessNext_g, bufferSize));
  checkCudaErrors(cudaMalloc(&d_imageGuessNext_b, bufferSize));

  initBuffer<<<blocks, threads>>>(d_imageGuessPrev_r, d_sourceImg_r, numPixels);
  initBuffer<<<blocks, threads>>>(d_imageGuessPrev_g, d_sourceImg_g, numPixels);
  initBuffer<<<blocks, threads>>>(d_imageGuessPrev_b, d_sourceImg_b, numPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  ENDCLOCK(step_four);

  // step 5 - perform Jacobi iteration 800 times
  STARTCLOCK(step_five);

  const int numJacobiIterations = 800;

  float sum = 0.0f;
  int cnt = 0;
  float min = 1000;
  float max = 0.0f;

  for (int i = 0; i < numJacobiIterations; i++) {
    GpuTimer jacobiTimer;
    jacobiTimer.Start();

    jacobi<<<step2Blocks, step2Threads>>>(d_imageGuessPrev_r, d_imageGuessNext_r, d_interior, d_sourceImg_r, d_dstImg_r, numColsSource, numRowsSource);
    jacobi<<<step2Blocks, step2Threads>>>(d_imageGuessPrev_g, d_imageGuessNext_g, d_interior, d_sourceImg_g, d_dstImg_g, numColsSource, numRowsSource);
    jacobi<<<step2Blocks, step2Threads>>>(d_imageGuessPrev_b, d_imageGuessNext_b, d_interior, d_sourceImg_b, d_dstImg_b, numColsSource, numRowsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    std::swap(d_imageGuessPrev_r, d_imageGuessNext_r);
    std::swap(d_imageGuessPrev_g, d_imageGuessNext_g);
    std::swap(d_imageGuessPrev_b, d_imageGuessNext_b);

    jacobiTimer.Stop();

    float jacobiTime = jacobiTimer.Elapsed();
    sum += jacobiTime;
    cnt++;
    min = std::min(min, jacobiTime);
    max = std::max(max, jacobiTime);
  }

  std::swap(d_imageGuessPrev_r, d_imageGuessNext_r);    // results will be in image previous
  std::swap(d_imageGuessPrev_g, d_imageGuessNext_g);
  std::swap(d_imageGuessPrev_b, d_imageGuessNext_b);

  ENDCLOCK(step_five);

  printf("\tavg: %f, min: %f, max: %f\n", sum/cnt, min, max);

  // step 6 - create output image
  STARTCLOCK(step_six);

  finalizeJacobi<<<blocks, threads>>>(d_imageGuessPrev_r, d_sourceImg_r, d_dstImg_r, d_interior, numPixels);
  finalizeJacobi<<<blocks, threads>>>(d_imageGuessPrev_g, d_sourceImg_g, d_dstImg_g, d_interior, numPixels);
  finalizeJacobi<<<blocks, threads>>>(d_imageGuessPrev_b, d_sourceImg_b, d_dstImg_b, d_interior, numPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  ENDCLOCK(step_six);

  /***
  optimizations
  reduce to subset of image that only includes stuff in the border and inter
      find maxX, minX, maxY, minY
      create new image with given boundaries
      then do jacobi iterations
  combine imgGuessPrev and dstImg into single array to reduce branching and mem access
  use bitstring for interior

  */

  // book keeping
  STARTCLOCK(bookKeeping);

  recombineChannels<<<blocks, threads>>>(d_blendedImg, d_dstImg_r, d_dstImg_g, d_dstImg_b, numPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, imageSize, cudaMemcpyDeviceToHost));  // copy results to host

  ENDCLOCK(bookKeeping);
  
  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */



  /* The reference calculation is provided below, feel free to use it
     for debugging purposes. 
   */

  /*
    uchar4* h_reference = new uchar4[srcSize];
    reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
    delete[] h_reference; */
}
