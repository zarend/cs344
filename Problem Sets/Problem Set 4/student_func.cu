//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

#include <thrust/scan.h>
#include <stdio.h>
#include <assert.h>
#include "timer.h"

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

#define PROFILE 0

#if PROFILE
  #define STARTCLOCK(name) GpuTimer name; name.Start();
  #define ENDCLOCK(name) name.Stop(); printf("%s: %f\n", #name, name.Elapsed());
#else
  #define STARTCLOCK(name) 
  #define ENDCLOCK(name) 
#endif

void endClock(clock_t start, char message[]) {
  clock_t end = clock();
  double elapsed_secs = end-start / (double)CLOCKS_PER_SEC;
  printf("start: %d, end: %d\n", start, end);
  printf("[mytimer] %s: %.3lf msec\n", message, elapsed_secs * 1000.0);
}

__global__
void histo(unsigned int * d_vals_src, unsigned int * d_histo, int bit, unsigned int mask, size_t numElems, int numBins) {
    extern __shared__ uint sharedHisto[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems) {
        if (threadIdx.x < numBins) {
            sharedHisto[threadIdx.x] = 0;
        }
        __syncthreads();

        int bin = (d_vals_src[idx] & mask) >> bit;
        atomicAdd(sharedHisto + bin, 1);
        __syncthreads();

        if (threadIdx.x < numBins) {
            atomicAdd(d_histo + threadIdx.x, sharedHisto[threadIdx.x]);
        }
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
void convertToExclusive(uint * arr, uint * blockEnds, size_t length) {    // move everything over one and set 0th element to additive identity
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        if (threadIdx.x) {
            uint temp = idx ? arr[idx - 1] : 0;
            if (threadIdx.x == blockDim.x - 1) {
                blockEnds[blockIdx.x] = arr[idx];
            }

            
            __syncthreads();
            arr[idx] = temp;
        }
    }
}

__global__
void convertToExclusivePass2(uint * arr, uint * blockEnds, size_t length, int threads) {    // move everything over one and set 0th element to additive identity
    uint idx = threadIdx.x * threads;
    if (idx && idx < length) {
        arr[idx] = blockEnds[threadIdx.x - 1];
    }
}

__global__
void inclusiveScanPass1(uint * arr, uint * blockSum, size_t length) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        for (uint offset = 1; offset < blockDim.x && threadIdx.x >= offset; offset *= 2) {
            uint temp = arr[idx - offset];
            __syncthreads();
            arr[idx] += temp;
            __syncthreads();
        }

        __syncthreads();

        if (threadIdx.x == blockDim.x - 1) {
            blockSum[blockIdx.x] = arr[idx];
        }
    }
}

void checkPt1(uint * d_arr, size_t length) {
    uint * h_arr = (uint *)malloc(sizeof(uint) * length);
    checkCudaErrors(cudaMemcpy(h_arr, d_arr, sizeof(uint) * length, cudaMemcpyDeviceToHost));

    for (int i = 1; i < length; i++) {
        if (i % 1024) {
            if (h_arr[i] < h_arr[i - 1]) {
                printf("i: %d, h_arr[i]: %u, h_arr[i-1]: %u\n", i, h_arr[i], h_arr[i - 1]);
            }
            assert(h_arr[i] >= h_arr[i - 1]);
        }
    }
}

__global__
void inclusiveScanPass2(uint * arr, uint * blockSum, size_t length) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        arr[idx] += blockSum[blockIdx.x];
    }
}

void exclusiveSumDwarf(uint * d_arr, size_t length) {   // takes the exclusive prefix sum of d_arr
    const int threads = MAX_THREADS;
    const int blocks = length / threads + 1;

    STARTCLOCK(______convertToExclusive);
    uint * d_blockEnds;
    checkCudaErrors(cudaMalloc(&d_blockEnds, blocks * sizeof(uint)));
    convertToExclusive<<<blocks, threads>>>(d_arr, d_blockEnds, length);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    convertToExclusivePass2<<<1, blocks>>>(d_arr, d_blockEnds, length, threads);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    /*uint * d_temp;  // shift everything over by one and set 0th element to 0
    checkCudaErrors(cudaMalloc(&d_temp, length * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_temp, d_arr, length * sizeof(uint), cudaMemcpyDeviceToDevice));
    convertToExclusive<<<blocks, threads>>>(d_temp, d_arr, length);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_temp));*/
    ENDCLOCK(______convertToExclusive);

    /*uint * h_thrustArr = (uint *)malloc(length * sizeof(uint));
    uint * h_arr = (uint *)malloc(length * sizeof(uint));
    checkCudaErrors(cudaMemcpy(h_thrustArr, d_arr, length * sizeof(uint), cudaMemcpyDeviceToHost));*/

    STARTCLOCK(______inclusivScanPass1); //take inclusive sum of each block and write out each blocks sum
    uint * d_blockSum;
    checkCudaErrors(cudaMalloc(&d_blockSum, blocks * sizeof(uint)));
    inclusiveScanPass1<<<blocks, threads>>>(d_arr, d_blockSum, length);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    ENDCLOCK(______inclusivScanPass1);

    STARTCLOCK(______exclusiveSum); // take exclusive sum of the sums for each block
    exclusiveSum<<<1, blocks>>>(d_blockSum, blocks);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    ENDCLOCK(______exclusiveSum);

    STARTCLOCK(______pass2); // add in those sums for each block
    inclusiveScanPass2<<<blocks, threads>>>(d_arr, d_blockSum, length);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    ENDCLOCK(______pass2);

    // take thrust inclusive sum
    /*thrust::inclusive_scan(h_thrustArr, h_thrustArr + length, h_thrustArr);
    checkCudaErrors(cudaMemcpy(h_arr, d_arr, length * sizeof(uint), cudaMemcpyDeviceToHost));

    int diff = 0;
    int numPrints = 0;
    for (int i = 0; i < length; i++) {
        if (h_arr[i] != h_thrustArr[i]) {
            diff = 1;
            numPrints++;
            printf("h_arr[%d]: %u, h_thrustArr[%d]: %u\n", i, h_arr[i], i, h_thrustArr[i]);
            if (numPrints > 100) {
                exit(0);
            }
        }
    }*/

}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    const int numBits = 4;
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
        STARTCLOCK(singlePass);
        unsigned int mask = (numBins - 1) << bit;

        checkCudaErrors(cudaMemset(d_histo, 0, numBins * sizeof(unsigned int)));


        const int threadsPerBlock = MAX_THREADS;  // generate histogram
        const int numBlocks = numElems / threadsPerBlock + 1;

        // 72ms
        STARTCLOCK(__histoClock);
        #if 1
            size_t sharedSize = numBins * sizeof(uint);

            histo<<<numBlocks*4, threadsPerBlock/4, sharedSize>>>(d_vals_src, d_histo, bit, mask, numElems, numBins);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        #endif
        ENDCLOCK(__histoClock);

        STARTCLOCK(__exclusiveSumClock);
        // 17 ms
        exclusiveSum<<<1, numBins>>>(d_histo, numBins); // take exclusive prefix sum of histogram
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        ENDCLOCK(__exclusiveSumClock);

        STARTCLOCK(__scatterClock);
        for (int binIdx = 0; binIdx < numBins; binIdx++) {  // for each bin
            STARTCLOCK(____initIndiciesKernel); // setup indices for exclusive sum scan
            initIndicies<<<numBlocks, threadsPerBlock>>>(d_outIdx, numElems, mask, bit, d_vals_src, binIdx);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
            ENDCLOCK(____initIndiciesKernel);

            STARTCLOCK(____exclusiveSumDwarf);
            exclusiveSumDwarf(d_outIdx, numElems);
            ENDCLOCK(____exclusiveSumDwarf);
            //exclusiveSumElf(d_outIdx, numElems);
            //checkCudaErrors(cudaMemcpy(h_outIdx, d_outIdx, numElems * sizeof(uint), cudaMemcpyDeviceToHost));
            //thrust::exclusive_scan(h_outIdx, h_outIdx + numElems, h_outIdx);
            //checkCudaErrors(cudaMemcpy(d_outIdx, h_outIdx, numElems * sizeof(uint), cudaMemcpyHostToDevice));

            STARTCLOCK(____addStuffKernel);
            addStuff<<<numBlocks, threadsPerBlock>>>(d_outIdx, d_histo, binIdx, numElems);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
            ENDCLOCK(____addStuffKernel);

            STARTCLOCK(____scatterKernel);
            scatter<<<numBlocks, threadsPerBlock>>>(d_vals_src, d_vals_dst, d_pos_src, d_pos_dst, d_outIdx, mask, bit, d_histo, binIdx, numElems);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
            ENDCLOCK(____scatterKernel);
        }
        ENDCLOCK(__scatterClock);

        // cleanup
        std::swap(d_vals_src, d_vals_dst);
        std::swap(d_pos_src, d_pos_dst);
        ENDCLOCK(singlePass);

        #if PROFILE
            printf("\n");
        #endif
    }

    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}
