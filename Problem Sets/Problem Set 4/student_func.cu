//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

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

#define BITS_PER_BYTE 8
#define MAX_THREADS 1024

__global__
void histo(unsigned int * d_vals_src, unsigned int * d_histo, int bit, unsigned int mask, size_t numElems) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems) {
        int bin = (d_vals_src[idx] & mask) >> bit;
        atomicAdd(d_histo + bin, 1);
    }
}

__global__
void exclusiveSum(unsigned int * histo, int numBins) {
    unsigned int idx = threadIdx.x;

    if (idx < numBins) {
        unsigned int temp = idx ? histo[idx - 1] : 0; //convert from inclusive to exclusive scan
        __syncthreads();
        histo[idx] = temp;
        __syncthreads();

        for (int offset = 1; offset < numBins && idx >= offset; offset *= 2) {
            histo[idx] += histo[idx - offset];
            __syncthreads();
        }
    }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    const int numBits = 1;
    const int numBins = 1 << numBits;

    unsigned int * d_histo;
    checkCudaErrors(cudaMalloc(&d_histo, numBins * sizeof(unsigned int)));

    unsigned int * d_vals_src = d_inputVals;
    unsigned int * d_pos_src = d_inputPos;

    unsigned int * d_vals_dst = d_outputVals;
    unsigned int * d_pos_dst = d_outputPos;

    for (unsigned int bit = 0; bit < BITS_PER_BYTE * sizeof(unsigned int); bit += numBits) {
        unsigned int mask = (numBins - 1) << bit;

        checkCudaErrors(cudaMemset(d_histo, 0, numBins * sizeof(unsigned int)));

        int threadsPerBlock = MAX_THREADS;  // generate histogram
        int numBlocks = numElems / threadsPerBlock + 1;

        histo<<<numBlocks, threadsPerBlock>>>(d_vals_src, d_histo, bit, mask, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        exclusiveSum<<<1, numBins>>>(d_histo, numBins); // take exclusive scan of histogram
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // scatter
        size_t size = numElems * sizeof(unsigned int);
        unsigned int * h_vals_src = (unsigned int *)malloc(size);
        unsigned int * h_pos_src = (unsigned int *)malloc(size);
        unsigned int * h_vals_dst = (unsigned int *)malloc(size);
        unsigned int * h_pos_dst = (unsigned int *)malloc(size);
        unsigned int * h_histo = (unsigned int *)malloc(numBins * sizeof(unsigned int));

        checkCudaErrors(cudaMemcpy(h_vals_src, d_vals_src, size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_pos_src, d_pos_src, size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_vals_dst, d_vals_dst, size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_pos_dst, d_pos_dst, size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_histo, d_histo, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        for (unsigned int j = 0; j < numElems; ++j) {
          unsigned int bin = (h_vals_src[j] & mask) >> bit;
          h_vals_dst[h_histo[bin]] = h_vals_src[j];
          h_pos_dst[h_histo[bin]]  = h_pos_src[j];
          h_histo[bin]++;
        }

        checkCudaErrors(cudaMemcpy(d_vals_src, h_vals_src, size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_pos_src, h_pos_src, size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_vals_dst, h_vals_dst, size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_pos_dst, h_pos_dst, size, cudaMemcpyHostToDevice));

        // cleanup
        std::swap(d_vals_src, d_vals_dst);
        std::swap(d_pos_src, d_pos_dst);
    }

    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}
