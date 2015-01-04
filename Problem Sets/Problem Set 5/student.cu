/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

#define MAX_THREADS_PER_BLOCK 1024

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= 0 && idx < numVals) {
    int bin = vals[idx];
    atomicAdd(histo + bin, 1);
  }
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  const dim3 blockSize(MAX_THREADS_PER_BLOCK, 1, 1);
  const dim3 gridSize(numElems / blockSize.x + 1);

  yourHisto<<<gridSize, blockSize>>>(d_vals, d_histo, numElems);
}
