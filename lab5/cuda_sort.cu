// Made by Max Bronnikov
#include <iostream>
#include <vector>
#include <stdexcept>
#include <chrono>

#define INT_LIMIT 16777216 // 2^24
// #define INT_LIMIT 64 // 2^24
// #define THREADS 16
// #define THREADS_X2 32
#define MAX_BLOCKS 1024u
#define THREADS 512u
#define THREADS_X2 1024u

#define NUM_BANKS 16u
#define LOG_NUM_BANKS 4u

// This offset definition helps to avoid bank conflicts in scan
// ref: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#define AVOID_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (LOG_NUM_BANKS << 1))


#define RELEASE // TIME_COUNT // set for count time 
#define SORT_GPU // set device for computing

using namespace std;
using namespace std::chrono;

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

    // Block E: 
    // Write results back into global mem
    // We use '+=' instead '=' because we want get inclusive scan(standart algo gives exclusive)
    __syncthreads();
    data[ai] += shared_temp[ai + bank_offset_a];
    data[bi] += shared_temp[bi + bank_offset_b];
}

// This method runs block scans with step determined by count of runned blocks
// We think, that size % THREADS_X2 = 0 (it requirement to allocation procedure from host) 
__global__
void scan_step(uint32_t* data_in, const uint32_t size){
    __shared__ uint32_t temp[THREADS_X2 * sizeof(uint32_t)];

    const uint32_t thread_id = threadIdx.x;
    const uint32_t start = blockIdx.x * THREADS_X2;
    const uint32_t step = THREADS_X2 * gridDim.x;

    for(uint32_t offset = start; offset < size; offset += step){
        // launch scan algo for block
        block_scan(thread_id, &data_in[offset], temp, THREADS_X2);
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
        data[i*THREADS_X2 + tid + tid] += add; // no memory conflicts here
        data[i*THREADS_X2 + tid + tid + 1] += add; // no memory conflicts here
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
    uint32_t blocks = size / THREADS_X2; // each block computes 2*THREADS values
    uint32_t launch_blocks = blocks > MAX_BLOCKS ? MAX_BLOCKS : blocks;

    scan_step<<<launch_blocks, THREADS>>>(d_data, size); // launch scan per block kernel
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel
    cudaThreadSynchronize(); // wait end

    // condition for stop of recursia
    if(blocks == 1){
        return;
    }

    // alloc new data wit overheap for correct scan on next recursive step
    uint32_t* addition_sums = nullptr;
    uint32_t alloc_size = compute_offset(blocks, THREADS_X2); // overheap
    throw_on_cuda_error(cudaMalloc((void**)&addition_sums, alloc_size*sizeof(uint32_t))); 

    // Copy mem with stride:
    throw_on_cuda_error(cudaMemcpy2D(
        addition_sums, 
        sizeof(uint32_t), 
        d_data + (THREADS_X2 - 1), // get last elem of each block
        THREADS_X2 * sizeof(uint32_t), // stride between values
        sizeof(uint32_t), blocks, // copy #blocks elements
        cudaMemcpyDeviceToDevice
    ));

    // launch recursia for sums:
    scan(addition_sums, alloc_size);

    // add sums to native data:
    add_block_sums<<<launch_blocks, THREADS>>>(d_data, addition_sums, blocks);
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

    // No need sync because cudaFree will do it

    // free useless data:
    throw_on_cuda_error(cudaFree(addition_sums));
}

// count sort on gpu inplace
void cuda_count_sort(int32_t* h_data, const uint32_t size){
    // device data:
    uint32_t* d_counts;
    int32_t* d_data;

    // alloc data with overheap for scan algo
    throw_on_cuda_error(cudaMalloc((void**)&d_data, size*sizeof(int32_t)));
    throw_on_cuda_error(cudaMalloc(
        (void**)&d_counts, 
        compute_offset(INT_LIMIT, THREADS_X2)*sizeof(uint32_t))
    );
    
    // copy data into buffers
    throw_on_cuda_error(cudaMemcpy(d_data, h_data, sizeof(int32_t)*size, cudaMemcpyHostToDevice));
    throw_on_cuda_error(cudaMemset(d_counts, 0, INT_LIMIT*sizeof(uint32_t))); // init histogram with zero

    // step 1: compute histogram
    compute_histogram<<<MAX_BLOCKS, THREADS>>>(d_counts, d_data, size);
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel
    cudaThreadSynchronize(); // wait end

    // step 2: prefix sums(inclusive scan) of histogram
    scan(d_counts, INT_LIMIT);

    // step 3: change input to true order
    sort_by_counts<<<MAX_BLOCKS, THREADS>>>(d_data, d_counts, INT_LIMIT);
    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

    // copy data back:
    throw_on_cuda_error(cudaMemcpy(h_data, d_data, sizeof(int32_t)*size, cudaMemcpyDeviceToHost));

    

    // Free data:
    throw_on_cuda_error(cudaFree(d_data));
    throw_on_cuda_error(cudaFree(d_counts));
}


// count sort on cpu inplace
void cpu_count_sort(int32_t* data, uint32_t size){
    if(!size){
        return;
    }

    uint32_t* counter = new uint32_t[INT_LIMIT];
    memset(counter, 0, INT_LIMIT*sizeof(uint32_t));

    // histogram
    for(uint32_t j = 0; j < size; j++){
        ++counter[data[j]];
    }

    // exclusive scan
    uint32_t sum = 0;
    for(uint32_t j = 0; j < INT_LIMIT; ++j){
        uint32_t temp = counter[j];
        counter[j] = sum;
        sum += temp;
    }

    // Sorting:

    // without last element
    for(uint32_t j = 0; j < INT_LIMIT - 1; ++j){
        while(counter[j] < counter[j + 1]){
            data[counter[j]++] = j;
        }
    }

    // last element
    {
        while(counter[INT_LIMIT - 1] < size){
            data[counter[INT_LIMIT - 1]++] = INT_LIMIT - 1;
        }  
    }

    delete[] counter;
}

/*
==========================================================================================
==================== END COUNT SORT ======================================================
==========================================================================================
*/

// function for read data from stream
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


    try
    {
        // Launch cuda sort

        #ifdef TIME_COUNT
        auto start = steady_clock::now();
        #endif

        #ifdef SORT_GPU
        cuda_count_sort(data, data_size);
        #else
        cpu_count_sort(data, data_size);
        #endif

        #ifdef TIME_COUNT
        auto end = steady_clock::now();
        cerr << "===============================" << endl;
        cerr << "INFERENCE TIME: ";
        cerr << ((double)duration_cast<microseconds>(end - start).count()) / 1000.0 << "ms" << endl;
        cerr << "===============================" << endl;
        #endif

    }catch(const std::runtime_error& err){
        cerr << err.what() << endl;
        delete[] data;
        return 0;
    }
    
    // write data and clear buffer
    cout.write(reinterpret_cast<const char*>(data), data_size*sizeof(int32_t));

    delete[] data;

    return 0;
}