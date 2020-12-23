// Made by Max Bronnikov
#include <iostream>
#include <vector>

#define INT_LIMIT 16777216 // 2^24
#define MAX_BLOCKS 1024
#define THREADS 512

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// This offset definition helps to avoid bank conflicts in scan
// ref: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#define AVOID_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (LOG_NUM_BANKS << 1))

using namespace std;

// Error handler
void throw_on_cuda_error(const cudaError_t& code)
{
  if(code != cudaSuccess)
  {
    string err_str = "CUDA Error: ";
    err_str += cudaGetErrorString(code);
    throw std::runtime_error(err_str);
  }
}


///////////////////////////////////////////////////////////////////////
//////////////////////// DEVICE CODE //////////////////////////////////
///////////////////////////////////////////////////////////////////////

// Note: We can compute histogram without shared memory, because 
//       shared mem only 16Kb per block and we cant compute effective in shared mem.
//       Also, a lot destribution of values helps to avoid memory access conflicts in general.
__global__ 
void compute_histogram(uint32_t* histogram, const int32_t* data, const uint32_t data_size){
    uint32_t step = blockDim.x * gridDim.x;
    uint32_t start = threadIdx.x + blockIdx.x*blockDim.x;
    // compute statistic for each value with step
    for(uint32_t i = start; i < data_size; i += step){
        atomicAdd(&histogram[data[i]], 1);
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

    // Write results back into global mem
    __syncthreads();

    // Block E:
    // We use '+=' instead '=' because we want get inclusive scan(standart algo gives exclusive)
    data[ai] += shared_temp[ai + bank_offset_a];
    data[bi] += shared_temp[bi + bank_offset_b];
}

// This method runs block scans with step determined by count of runned blocks
// We think, that size % THREADS = 0 (it requirement to allocation procedure from host) 
__global__
void scan_step(uint32_t* data_in, const uint32_t size){
    __shared__ uint32_t temp[(THREADS * sizeof(uint32_t)) << 1];

    const uint32_t thread_id = threadIdx.x;
    const uint32_t start = blockIdx.x * blockDim.x;
    const uint32_t step = blockDim.x * gridDim.x;

    for(uint32_t offset = start; offset < size; offset += step){
        // launch scan algo for block
        block_scan(thread_id, &data_in[offset], temp, THREADS);
    }
}

// Important: Launch this kernel with same thread count as scan_step.
__global__
void add_block_sums(uint32_t* data, const uint32_t* sums, const uint32_t sums_count){
    uint32_t start = blockIdx.x + 1;
    uint32_t step = gridDim.x;
    uint32_t tid = threadIdx.x;
    uint32_t block_size = blockDim.x;
    
    for(uint32_t i = start; i < sums_count; i += step){
        data[i*block_size + tid] = sums[i - 1]; // no memory conflicts here
    }
}

// sort with precomputed data 
// ref: https://www.researchgate.net/publication/245542734
__global__
void sort_by_counts(int32_t* data, const uint32_t* counts, const uint32_t c_size){
    int32_t idx = blockDim.x * blockIdx.x +  threadIdx.x;
    int32_t step = blockDim.x * gridDim.x;

    for(int32_t tid = idx; tid < c_size; tid += step){
        int32_t low = tid ? counts[tid - 1] : 0;

        for(int32_t i = counts[tid] - 1; i >= low; --i){
            data[i] = tid;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
///////////////////// END DEVICE CODE ///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////


/*
==============================================================================
===================== COUNT SORT =============================================
==============================================================================
*/


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
    uint32_t blocks = size / THREADS;
    uint32_t launch_blocks = blocks > MAX_BLOCKS ? MAX_BLOCKS : blocks;

    scan_step<<<launch_blocks, THREADS>>>(d_data, size); // launch scan per block kernel
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

    // No need synchronize because cudaMalloc bellow will do it instead us :)

    // condition for stop of recursia
    if(blocks == 1){
        return;
    }

    // alloc new data wit overheap for correct scan on next recursive step
    uint32_t* addition_sums = nullptr;
    uint32_t alloc_size = compute_offset(blocks, THREADS);
    throw_on_cuda_error(cudaMalloc((void**)&addition_sums, alloc_size*sizeof(uint32_t))); 

    // Copy mem with stride:
    throw_on_cuda_error(cudaMemcpy2D(
        addition_sums, 
        sizeof(uint32_t), 
        d_data + (THREADS - 1), // get last elem of each block
        THREADS * sizeof(uint32_t), // stride between values
        sizeof(uint32_t), blocks, // copy #blocks elements
        cudaMemcpyDeviceToDevice
    ));

    // launch recursia:
    scan(addition_sums, alloc_size);

    // add sums to native data:
    add_block_sums<<<launch_blocks, THREADS>>>(d_data, addition_sums, blocks);
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

    // free useless data:
    cudaFree(addition_sums);
}

void inplace_count_sort(int32_t* h_data, const uint32_t size){
    // device data:
    uint32_t* d_counts;
    int32_t* d_data;

    // alloc data with overheap for scan algo
    throw_on_cuda_error(cudaMalloc((void**)&d_data, size*sizeof(int32_t)));
    throw_on_cuda_error(cudaMalloc((void**)&d_counts, compute_offset(INT_LIMIT, THREADS)*sizeof(uint32_t)));
    
    // copy data into buffers
    throw_on_cuda_error(cudaMemcpy(d_data, h_data, sizeof(int32_t)*size, cudaMemcpyHostToDevice));
    throw_on_cuda_error(cudaMemset(d_counts, 0, INT_LIMIT*sizeof(uint32_t))); // init histogram with zero

    // step 1: compute histogram
    compute_histogram<<<MAX_BLOCKS, THREADS>>>(d_counts, d_data, size);
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel
    cudaThreadSynchronize(); // wait end

    // step 2: prefix sums of histogram
    scan(d_counts, INT_LIMIT);
    cudaThreadSynchronize(); // wait end

    // step 3: change input to true order
    sort_by_counts<<<MAX_BLOCKS, THREADS>>>(d_data, d_counts, INT_LIMIT);
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

    // copy data back:
    throw_on_cuda_error(cudaMemcpy(h_data, d_data, sizeof(int32_t)*size, cudaMemcpyDeviceToHost));

    // Free data:
    throw_on_cuda_error(cudaFree(d_data));
    throw_on_cuda_error(cudaFree(d_counts));
}

/*
==========================================================================================
==================== END COUNT SORT ======================================================
==========================================================================================
*/

int32_t* read_data(uint32_t& data_size, istream& is){
    // host data:
    int32_t* data = nullptr;

    // read data size
    is.read(reinterpret_cast<char*>(&data_size), sizeof(uint32_t));

    data = new int32_t[data_size];

    if(not data){
        throw std::runtime_error("Wrong data alloc!");
    }
    
    // read and copy data into buffers
    is.read(reinterpret_cast<char*>(data), data_size*sizeof(int32_t));

    return data;
}

int main(){
    // host data:
    uint32_t data_size = 0;
    int32_t* data = nullptr;
    // read data from cin
    try{
        data = read_data(data_size, cin);
    }catch(std::runtime_error& err) {
        cerr << err.what() << endl;
        return 0;
    }

    // sort data
    try
    {
        inplace_count_sort(data, data_size);
    }catch(const std::runtime_error& err){
        cerr << err.what() << endl;
    }
    
    // write data and clear buffer
    cout.write(reinterpret_cast<const char*>(data), data_size*sizeof(int32_t));

    delete[] data;

    return 0;
}