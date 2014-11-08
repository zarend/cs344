//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

#include <thrust/scan.h>

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

#define uint unsigned int

__global__
void histo(unsigned int * d_vals_src, unsigned int * d_histo, int bit, unsigned int mask, size_t numElems) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems) {
        int bin = (d_vals_src[idx] & mask) >> bit;
        atomicAdd(d_histo + bin, 1);
    }
}

__global__
void scatter(uint * d_vals_src, uint * d_vals_dst, uint * d_pos_src, uint * d_pos_dst, uint * d_outIdx, uint mask, uint bit, uint* d_histo, int binIdx, size_t numElems) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems) {
        int bin = (d_vals_src[idx] & mask) >> bit;
        if (bin == binIdx) {
            d_vals_dst[d_outIdx[idx]] = d_vals_src[idx];
            d_pos_dst[d_outIdx[idx]] = d_pos_src[idx];
        }
    }
}

__global__
void addStuff(uint * d_outIdx, uint * d_histo, int binIdx, size_t numElems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems) {
        d_outIdx[idx] += d_histo[binIdx];
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

__global__
void initIndicies(unsigned int * d_outIdx, size_t numElems, unsigned int mask, unsigned int bit, unsigned int * d_vals_src, int bin) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems) {
        int binIdx = (d_vals_src[idx] & mask) >> bit;
        d_outIdx[idx] = binIdx == bin ? 1 : 0;
    }
}

__global__
void addInBins(unsigned int * d_outIdx, unsigned int * d_vals_src,unsigned int * d_histo, unsigned int bit, unsigned int mask, size_t numElems, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems) {
        int bin = (d_vals_src[idx] & mask) >> bit;
        d_outIdx[idx] += d_histo[bin];
    }
}

__global__
void finishScatter(unsigned int * d_vals_dst, unsigned int * d_vals_src, unsigned int * d_pos_dst, unsigned int * d_pos_src, unsigned int * d_outIdx, size_t numElems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems) {
        d_vals_dst[d_outIdx[idx]] = d_vals_src[idx];
        d_pos_dst[d_outIdx[idx]] = d_pos_src[idx];
    }
}

__global__
void inclusiveSumPass1(uint * d_arr, uint * d_out, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        for (int offset = 1; offset < blockDim.x && idx >= offset; offset *= 2) {
            d_arr[idx] += d_arr[idx - offset];
            __syncthreads();
        }

        if (threadIdx.x == blockDim.x - 1) {    // last tread in block writes
            d_out[blockIdx.x] = d_arr[idx];     // its result to d_out
        }
    }
}

__global__
void inclusiveSumPass2(uint * d_arr, uint * d_out, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        d_arr[idx] += d_out[blockIdx.x];
    }
}

__global__
void convertToExclusive(uint * d_in, uint* d_out, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        if (idx) {
            d_out[idx] = d_in[idx - 1];
        }
        else {
            d_out[idx] = 0;
        }
    }
}

void exclusiveSumElf(unsigned int * d_arr, size_t length) {
    unsigned int threads = MAX_THREADS;
    unsigned int blocks = length / threads + 1;

    uint * d_out;
    checkCudaErrors(cudaMalloc(&d_out, length * sizeof(uint)));

    inclusiveSumPass1<<<blocks, threads>>>(d_arr, d_out, length);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    exclusiveSum<<<1, blocks>>>(d_out, blocks);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    inclusiveSumPass2<<<blocks, threads>>>(d_arr, d_out, length);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    uint * temp;
    checkCudaErrors(cudaMalloc(&temp, length * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(temp, d_arr, length * sizeof(uint), cudaMemcpyDeviceToDevice));

    convertToExclusive<<<blocks, threads>>>(temp, d_arr, length);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    /*plan:
    take inclusive sum scan for each block
    then write out last element of each block to out array
    take inclusive sum scan on out array
    map out array to each block in original array
    convert inclusive sum to exclusive by moving everything over one and settings
        element 0 to 0
    */
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

    unsigned int * d_outIdx;
    checkCudaErrors(cudaMalloc(&d_outIdx, numElems * sizeof(unsigned int)));

    uint * h_outIdx = (uint *)malloc(numElems * sizeof(uint));

    for (unsigned int bit = 0; bit < BITS_PER_BYTE * sizeof(unsigned int); bit += numBits) {
        unsigned int mask = (numBins - 1) << bit;

        checkCudaErrors(cudaMemset(d_histo, 0, numBins * sizeof(unsigned int)));

        const int threadsPerBlock = MAX_THREADS;  // generate histogram
        const int numBlocks = numElems / threadsPerBlock + 1;

        histo<<<numBlocks, threadsPerBlock>>>(d_vals_src, d_histo, bit, mask, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        exclusiveSum<<<1, numBins>>>(d_histo, numBins); // take exclusive prefix sum of histogram
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // scatter
        for (int binIdx = 0; binIdx < numBins; binIdx++) {  // for each bin
            // setup indices for exclusive sum scan
            initIndicies<<<numBlocks, threadsPerBlock>>>(d_outIdx, numElems, mask, bit, d_vals_src, binIdx);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

            //exclusiveSumElf(d_outIdx, numElems);
            checkCudaErrors(cudaMemcpy(h_outIdx, d_outIdx, numElems * sizeof(uint), cudaMemcpyDeviceToHost));
            thrust::exclusive_scan(h_outIdx, h_outIdx + numElems, h_outIdx);
            checkCudaErrors(cudaMemcpy(d_outIdx, h_outIdx, numElems * sizeof(uint), cudaMemcpyHostToDevice));

            addStuff<<<numBlocks, threadsPerBlock>>>(d_outIdx, d_histo, binIdx, numElems);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

            scatter<<<numBlocks, threadsPerBlock>>>(d_vals_src, d_vals_dst, d_pos_src, d_pos_dst, d_outIdx, mask, bit, d_histo, binIdx, numElems);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        }

        #if 0   // serial scatter implementation
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
        #endif

        // cleanup
        std::swap(d_vals_src, d_vals_dst);
        std::swap(d_pos_src, d_pos_dst);
    }

    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}
