// Made by Max Bronnikov
#ifndef __RAY_CLEANER__
#define __RAY_CLEANER__
#include "structures.cuh"
#include <vector>

#define CLEAN_BLOCKS 1024u
#define CLEAN_THREADS 512u
#define CLEAN_THREADS_X2 1024u

#define NUM_BANKS 16u
#define LOG_NUM_BANKS 4u

// This offset definition helps to avoid bank conflicts in scan
// ref: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#define AVOID_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (LOG_NUM_BANKS << 1))


using namespace std;

///////////////////////////////////////////////////////////////////////
//////////////////////// DEVICE CODE //////////////////////////////////
///////////////////////////////////////////////////////////////////////
__global__ 
void compute_binary(uint32_t* bins, const recursion* data, const uint32_t data_size, float boarder){
    uint32_t step = blockDim.x * gridDim.x;
    uint32_t start = threadIdx.x + blockIdx.x*blockDim.x;
    // compute statistic for each value with step
    for(uint32_t i = start; i < data_size; i += step){
        bins[i] = data[i].power < boarder ? 0 : 1;
    }
}


// Blelloch scan code (ref: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
// Note: This realization of scan avoid Bank conflcts, but implement scan only in one block. 
//       It means, that we need use 'scan and prpagate' technique for implement scan for arbitrary len recuresively.
__device__
void block_scan(const uint32_t tid, uint32_t* data, uint32_t* shared_temp, const uint32_t size){
    uint32_t ai = tid, bi = tid + (size >> 1);

    // Block A:
    uint32_t bank_offset_a = AVOID_OFFSET(ai);
    uint32_t bank_offset_b = AVOID_OFFSET(bi);
    uint32_t offset = 1;

    shared_temp[ai + bank_offset_a] = data[ai];
    shared_temp[bi + bank_offset_b] = data[bi];

    // up-sweep pass
    for(uint32_t d = size >> 1; d > 0; d >>= 1, offset <<= 1){
        __syncthreads(); // before next change of shared mem we need to sync last access

        if(tid < d){
            // Block B:
            uint32_t a_idx = offset*((tid << 1) + 1) - 1;
            uint32_t b_idx = offset*((tid << 1) + 2) - 1;

            a_idx += AVOID_OFFSET(a_idx);
            b_idx += AVOID_OFFSET(b_idx);

            shared_temp[b_idx] += shared_temp[a_idx];
        }
    }

    // Block C:
    if(!tid){
        shared_temp[size - 1 + AVOID_OFFSET(size - 1)] = 0; // set last elem zeros(first step of descent)
    }

    offset >>= 1;
    // down-sweep pass
    for(uint32_t d = 1; d < size; d <<= 1, offset >>= 1){
        __syncthreads(); // before next change of shared mem we need to sync last access
        if(tid < d){
            // Block D:
            uint32_t a_idx = offset*((tid << 1) + 1) - 1;
            uint32_t b_idx = offset*((tid << 1) + 2) - 1;

            a_idx += AVOID_OFFSET(a_idx);
            b_idx += AVOID_OFFSET(b_idx);

            uint32_t t = shared_temp[a_idx];
            shared_temp[a_idx] = shared_temp[b_idx];
            shared_temp[b_idx] += t;
        }
    }

    // Block E: 
    // Write results back into global mem
    // We use '+=' instead '=' because we want get inclusive scan(standart algo gives exclusive)
    __syncthreads();
    data[ai] += shared_temp[ai + bank_offset_a];
    data[bi] += shared_temp[bi + bank_offset_b];
}

// This method runs block scans with step determined by count of runned blocks
// We think, that size % CLEAN_THREADS_X2 = 0 (it requirement to allocation procedure from host) 
__global__
void scan_step(uint32_t* data_in, const uint32_t size){
    __shared__ uint32_t temp[CLEAN_THREADS_X2];

    const uint32_t thread_id = threadIdx.x;
    const uint32_t start = blockIdx.x * CLEAN_THREADS_X2;
    const uint32_t step = CLEAN_THREADS_X2 * gridDim.x;

    for(uint32_t offset = start; offset < size; offset += step){
        // launch scan algo for block
        block_scan(thread_id, &data_in[offset], temp, CLEAN_THREADS_X2);
    }
}

// Important: Launch this kernel with same thread count as scan_step.
__global__
void add_block_sums(uint32_t* data, const uint32_t* sums, const uint32_t sums_count){
    uint32_t start = blockIdx.x + 1;
    uint32_t step = gridDim.x;
    uint32_t tid = threadIdx.x;
    
    for(uint32_t i = start; i < sums_count; i += step){
        uint32_t add = sums[i - 1];
        data[i*CLEAN_THREADS_X2 + tid + tid] += add; // no memory conflicts here
        data[i*CLEAN_THREADS_X2 + tid + tid + 1] += add; // no memory conflicts here
    }
}

// TODO Optimize this to inplace sorting!
__global__
void sort_by_bins(
    recursion* rays, const recursion* copy, 
    const uint32_t* bins, const uint32_t* sums, 
    const uint32_t size
){
    int32_t idx = blockDim.x * blockIdx.x +  threadIdx.x;
    int32_t step = blockDim.x * gridDim.x;

    for(int32_t i = idx; i < size; i += step){
        if(bins[i]){
            rays[sums[i] - 1] = copy[i];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
///////////////////// END DEVICE CODE ///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////



// offset for allocation required by scan algo
uint32_t compute_offset(uint32_t size, const uint32_t base){
    if(size % base){
        size -= (size % base);
        size += base;
    }
    return size;
}

// Recursive function of scan for arbitrary lenght
void scan(uint32_t* d_data, const uint32_t size){
    uint32_t blocks = size / CLEAN_THREADS_X2; // each block computes 2*CLEAN_THREADS values
    uint32_t launch_blocks = blocks > CLEAN_BLOCKS ? CLEAN_BLOCKS : blocks;

    scan_step<<<launch_blocks, CLEAN_THREADS>>>(d_data, size); // launch scan per block kernel
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel
    cudaThreadSynchronize(); // wait end

    // condition for stop of recursia
    if(blocks == 1){
        return;
    }

    // alloc new data wit overheap for correct scan on next recursive step
    uint32_t* addition_sums = nullptr;
    uint32_t alloc_size = compute_offset(blocks, CLEAN_THREADS_X2); // overheap
    throw_on_cuda_error(cudaMalloc((void**)&addition_sums, alloc_size*sizeof(uint32_t))); 

    // Copy mem with stride:
    throw_on_cuda_error(cudaMemcpy2D(
        addition_sums, 
        sizeof(uint32_t), 
        d_data + (CLEAN_THREADS_X2 - 1), // get last elem of each block
        CLEAN_THREADS_X2 * sizeof(uint32_t), // stride between values
        sizeof(uint32_t), blocks, // copy #blocks elements
        cudaMemcpyDeviceToDevice
    ));

    // launch recursia for sums:
    scan(addition_sums, alloc_size);

    // add sums to native data:
    add_block_sums<<<launch_blocks, CLEAN_THREADS>>>(d_data, addition_sums, blocks);
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

    // No need sync because cudaFree will do it

    // free useless data:
    throw_on_cuda_error(cudaFree(addition_sums));
}


uint32_t cuda_clean_rays(recursion* d_rays, const uint32_t size, const float boarder = 0.005){
    if(!size){
        return 0;
    }

    uint32_t good_count = 0;
    
    uint32_t* d_bins;
    uint32_t* d_sums;
    recursion* d_rcopy;

    uint32_t help_data_size = compute_offset(size, CLEAN_THREADS_X2);
    
    throw_on_cuda_error(cudaMalloc(
        (void**)&d_bins, 
        size*sizeof(uint32_t))
    );
    throw_on_cuda_error(cudaMalloc(
        (void**)&d_sums, 
        help_data_size*sizeof(uint32_t))
    );
    throw_on_cuda_error(cudaMalloc(
        (void**)&d_rcopy,
        size*sizeof(recursion))
    );

    // step 0: init bins as 0
    throw_on_cuda_error(cudaMemset(d_bins, 0, size*sizeof(uint32_t))); 

    // step 1: compute binary set on gpu
    compute_binary<<<CLEAN_BLOCKS, CLEAN_THREADS>>>(d_bins, d_rays, size, boarder);
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel
    cudaThreadSynchronize(); // wait end

    // step 2: prefix sums(inclusive scan) of histogram
    // cpy bins data to sums for scan computing
    throw_on_cuda_error(cudaMemcpy(d_sums, d_bins, sizeof(int32_t)*size, cudaMemcpyDeviceToDevice));
    // get prefix sums of binary array
    scan(d_sums, help_data_size);
    // get count of `1` values from last element of prefix sums
    throw_on_cuda_error(cudaMemcpy(&good_count, d_sums + (size - 1), sizeof(int32_t), cudaMemcpyDeviceToHost));

    // step 3: move all good rays to start of array
    // copy all data to d_rcopy for inplace sort:
    throw_on_cuda_error(cudaMemcpy(d_rcopy, d_rays, sizeof(recursion)*size, cudaMemcpyDeviceToDevice));
    // "sort" bins here (strictly speaking, this is not really a sort)
    sort_by_bins<<<CLEAN_BLOCKS, CLEAN_THREADS>>>(d_rays, d_rcopy, d_bins, d_sums, size);
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

    // Free data:
    throw_on_cuda_error(cudaFree(d_bins));
    throw_on_cuda_error(cudaFree(d_sums));
    throw_on_cuda_error(cudaFree(d_rcopy));

    return good_count;
}


#endif