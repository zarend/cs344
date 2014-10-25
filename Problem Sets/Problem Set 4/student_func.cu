//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__
void histo(unsigned int* d_inputVals, unsigned int* d_histo, int startIdx, int endIdx, int bit) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int idx = myId + startIdx;

  if (idx < endIdx) {
    int bin = (d_inputVals[idx] & (1 << bit)) != 0;
    atomicAdd(d_histo + bin, 1);
  }
}

int sortHelper(unsigned int* d_inputVals, int startIdx, int endIdx, unsigned int* d_histo, int bit) {
  if (bit < 32) {
    int numBlocks = endIdx - startIdx;
    int numThreads = 1;

    cudaMemset(&d_histo, 0, 2*sizeof(unsigned int));
    histo<<<numBlocks, numThreads>>>(d_inputVals, d_histo, startIdx, endIdx, bit);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    int numZeroes;
    checkCudaErrors(cudaMemcpy(&numZeroes, d_histo, sizeof(int), cudaMemcpyDeviceToHost));

    
  }
  else {  // base case
  }

  return 0; //stub
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  const size_t size = numElems * sizeof(unsigned int);
  const int numBits = 32;

  unsigned int* d_histo;
  checkCudaErrors(cudaMalloc(&d_histo, 2 * sizeof(unsigned int)));

  for (int bit = 0; bit < numBits; bit++) {
    sortHelper(d_inputVals, 0, numElems, d_histo, 0);
  }
}
