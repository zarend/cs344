/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <cmath>
#include <stdio.h>

#define DEBUG 0
#define PROFILE 0

#define MAX(a, b) (a) > (b) ? (a) : (b)
#define MIN(a, b) (a) < (b) ? (a) : (b)

#define MAX_THREADS_PER_BLOCK 1024

enum opps {
  MIN,
  MAX
};

unsigned int nextPow2(unsigned int a) {
  double y = log2((double)a);
  int exponent = y + 0.5;

  return 1 << exponent;
}

#if PROFILE
  #define STARTCLOCK(name) clock_t (name) = startClock()
  #define ENDCLOCK(name) endClock(name, #name)
#else
  #define STARTCLOCK(name) 
  #define ENDCLOCK(name) 
#endif


clock_t startClock() {
  return clock();
}

void endClock(clock_t start, char message[]) {
  clock_t end = clock();
  double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
  printf("[mytimer] %s: %.3lf msec\n", message, elapsed_secs * 1000);
}

__global__
void reduce(float* d_in, float* d_out, int length, int opp) {
  extern __shared__ float sdata[];

  int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalIdx >= length) {
    return;
  }

  sdata[threadIdx.x] = d_in[globalIdx];
  __syncthreads();

  for (int offset = blockDim.x/2; offset && threadIdx.x < offset; offset /= 2) {
    if (globalIdx + offset < length) {
      float f1 = sdata[threadIdx.x];
      float f2 = sdata[threadIdx.x + offset];

      if (opp == MAX) {
        if (f2 > f1) {
          sdata[threadIdx.x] = f2;
        }
      }
      else if (opp == MIN) {
        if (f2 < f1) {
          sdata[threadIdx.x] = f1;
        }
      }
    }
    __syncthreads();
  }

  //d_out[blockIdx.x] = d_in[blockIdx.x * blockDim.x];

  if (threadIdx.x == 0) { // this block is a stub
    d_out[blockIdx.x] = sdata[threadIdx.x];
  }
}

__global__
void histofy(float* d_luminance, unsigned int *d_histo, int length, float lumMin, float lumRange, int numBins, int size) {
  extern __shared__ int shisto[];

  shisto[threadIdx.x] = 0;
  __syncthreads();

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length; idx += size) {
    int bin = (d_luminance[idx] - lumMin) / lumRange * numBins;

    if (bin < numBins) {
      atomicAdd(shisto + bin, 1);
    }
  }

  __syncthreads();
  
  if (threadIdx.x < numBins) {
    atomicAdd(d_histo + threadIdx.x, shisto[threadIdx.x]);
  }

}

__global__
void histoExclusiveScan(unsigned int *d_histo, int numBins) {
  int idx = threadIdx.x;

  // converts from inclusive to exclusive scan by moving over by one and setting 0th element to 0
  if (idx) {
    d_histo[idx] = d_histo[idx - 1];
  }
  else {
    d_histo[idx] = 0;
  }

  __syncthreads();

  for (int offset = 1; offset < numBins && idx >= offset; offset *= 2) {
    d_histo[idx] += d_histo[idx - offset];
    __syncthreads();
  }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  STARTCLOCK(wholeProgram);
  STARTCLOCK(part1);

  /*** #1 ***/
  int numCells = numRows * numCols;
  int size = sizeof(float) * numCells;

  float *d_luminance_cpy;   // make a copy of d_logLuminance
  checkCudaErrors(cudaMalloc(&d_luminance_cpy, size));
  checkCudaErrors(cudaMemcpy(d_luminance_cpy, d_logLuminance, size, cudaMemcpyDeviceToDevice));

  const dim3 blockSize(MAX_THREADS_PER_BLOCK, 1, 1);
  const dim3 gridSize(numCells/blockSize.x + 1, 1, 1);

  int sharedSize = blockSize.x * sizeof(float);

  float *round2Arr;
  checkCudaErrors(cudaMalloc(&round2Arr, sizeof(float)*gridSize.x));

  reduce<<<gridSize, blockSize, sharedSize>>>(d_luminance_cpy, round2Arr, numCells, MAX);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  reduce<<<gridSize, blockSize, sharedSize>>>(round2Arr, round2Arr, gridSize.x, MAX);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(&max_logLum, round2Arr, sizeof(float), cudaMemcpyDeviceToHost));

  // restore d_luminance_cpy
  checkCudaErrors(cudaMemcpy(d_luminance_cpy, d_logLuminance, size, cudaMemcpyDeviceToDevice));

  reduce<<<gridSize, blockSize, sharedSize>>>(d_luminance_cpy, round2Arr, numCells, MIN);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  reduce<<<gridSize, blockSize, sharedSize>>>(round2Arr, round2Arr, gridSize.x, MIN);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(&min_logLum, round2Arr, sizeof(float), cudaMemcpyDeviceToHost));

  #if DEBUG
    printf("numCells: %d, numBins: %d\n", numCells, numBins);
    printf("min_logLum: %f, max_logLum: %f\n", min_logLum, max_logLum); // should be (-4.0, 2.189105)
  #endif

  ENDCLOCK(part1);

  /*** #2***/
  STARTCLOCK(part2);
  float lumRange = max_logLum - min_logLum;
  ENDCLOCK(part2);

  /*** #3 ***/
  STARTCLOCK(part3);

  int histo_size = sizeof(int)*numBins;

  // restore d_luminance_cpy
  checkCudaErrors(cudaMemcpy(d_luminance_cpy, d_logLuminance, size, cudaMemcpyDeviceToDevice));

  int numBlocks = 56; // any more than 56 causes unspecified launch error
  int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;

  histofy<<<numBlocks, numThreadsPerBlock, histo_size>>>(d_luminance_cpy, d_cdf, numCells, min_logLum, lumRange, numBins, numBlocks * numThreadsPerBlock);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  ENDCLOCK(part3);

  /*** #4 ***/
  STARTCLOCK(part4);

  histoExclusiveScan<<<1, numBins>>>(d_cdf, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  ENDCLOCK(part4);
  ENDCLOCK(wholeProgram);
}
