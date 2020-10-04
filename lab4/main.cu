// Made by Max Bronnikov
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/swap.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "strided_range.h"
#include <iostream>
#include <map>
#include <string>
#include <iomanip>

using namespace std;
using namespace thrust;

const unsigned BLOCKS = 256;
const unsigned THREADS = 256;


void throw_on_cuda_error(const cudaError_t& code, int itter){
    if(code != cudaSuccess){
        string err = cudaGetErrorString(code);
        err += ", on iteration: ";
        err += to_string(itter);
        throw runtime_error(err);
    }
}

__global__ void gauss_step_L(double* C,  unsigned n, unsigned size, 
                                            unsigned col, double max_elem){
    unsigned thrd_idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned thrd_step = blockDim.x*gridDim.x;

    for(unsigned index = thrd_idx + col + 1; index < n; index += thrd_step){
        C[size*index + col] /= max_elem;
    }
}

__global__ void gauss_step_U(double* C, unsigned n, unsigned size, 
                                            unsigned col, double max_elem){
    unsigned thrd_idx = threadIdx.x;
    unsigned blck_idx = blockIdx.x;
    unsigned thrd_step = blockDim.x;
    unsigned blck_step = gridDim.x;

    unsigned starting_point_blck = col + 1;
    unsigned starting_point_thrd = (col + 1) >> 8; // start aligned by 256 

    
    for(unsigned i = blck_idx + starting_point_blck; i < n; i += blck_step){
        double coeff = C[i*size + col]; // get coeff

        // first itter may be not full
        unsigned j = thrd_idx + starting_point_thrd;

        if(thrd_idx + starting_point_thrd > col){
            C[i*size + j] -= coeff * C[col*size + j];
        }

        for(j += thrd_step; j < n; j+= thrd_step){
            C[i*size + j] -= coeff * C[col*size + j];
        }
    }
}

// 11111111 11111111 11111111 11111111
unsigned get_aligned_size(unsigned n){
    unsigned size = n;
    // 256 = 2^8 =>
    unsigned modulo = n & 255;
    if(modulo){
        size -= modulo;
        size += 256;
    }
    return size;
}

void swap_two_lines(device_vector<double>& matrix, unsigned idx, unsigned jdx, unsigned size){
    thrust::swap_ranges(thrust::device, 
        matrix.begin() + idx * size, 
        matrix.begin() + (idx + 1) * size, 
        matrix.begin() + jdx * size
    );
}


int main(){
    unsigned n;
    cin >> n;
    unsigned align = get_aligned_size(n);
    // alloc mem to union matrix(see wiki algorithm)
    host_vector<double> h_C(align * n);
    device_vector<double> d_C(align * n);
    host_vector<unsigned> h_ansvec(n);

    //host_vector<unsigned> h_p(n);
    //device_vector<unsigned> d_p(n);

    // 256 = 2^8 =>

    // input of matrix
    for(unsigned i = 0; i < n; ++i){
        h_ansvec[i] = i; // init of permutation vector
        for(unsigned j = 0; j < n; ++j){
            cin >> h_C[i*align + j]; 
        }
    }

    // transporting mem to device:
    d_C = h_C;

    // pointer to mem:
    double* raw_C = thrust::raw_pointer_cast(d_C.data());

    // compute  LU
    try{
        for(unsigned i = 0; i < n - 1; ++i){
            strided_range<thrust::device_vector<double>::iterator> 
                range(d_C.begin() + i, d_C.end(), align); // create iterator

            auto max_elem = thrust::max_element(range.begin() + i, range.end());
            unsigned max_idx = max_elem - range.begin();
            double max_val = *max_elem;

            cout << "Max elem: " << max_val << endl;

            if(max_idx != i){
                swap_two_lines(d_C, i, max_idx, align);
            }

            gauss_step_L<<<BLOCKS, THREADS>>>(raw_C, n, align, i, max_val);

            h_ansvec[i] = max_idx;

            throw_on_cuda_error(cudaGetLastError(), i);

            gauss_step_U<<<BLOCKS, THREADS>>>(raw_C, n, align, i, max_val);

            throw_on_cuda_error(cudaGetLastError(), i);
        }
    }catch(runtime_error& err){
        cout << "ERROR: " << err.what() << endl;
    }

    // memcpy from device to host
    h_C = d_C;

    // output for matrix:
    cout << std::scientific << std::setprecision(10);
    for(unsigned i = 0; i < n; ++i){
        for(unsigned j = 0; j < n; ++j){
            if(j){
                cout << " ";
            }
            cout << h_C[i*align + j];
        }
        cout << endl;
    }
    // output of vector
    for(unsigned i = 0; i < n; ++i){
        if(i){
            cout << " ";
        }
        cout << h_ansvec[i];
    }
    cout << endl;

    return 0;
}
