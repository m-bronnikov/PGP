// Made by Max Bronnikov
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/iterator/permutation_iterator.h>
#include <iostream>
#include <map>
#include <string>

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

__global__ void gauss_step(double* C, unsigned* p, unsigned n, unsigned col, double max_elem){
    unsigned thrd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned thrd_step = blockDim.x * gridDim.x;

    unsigned limit = n*n;
    for(unsigned index = thrd_idx; index < limit; index += thrd_step){
        unsigned j = index / n; // column in C
        unsigned real_i = index - n*j;
        unsigned virt_i = p[real_i]; // row in C defined by permutation matrix
    
        if(j < col || virt_i <= col){
            continue;
        }

        printf("[ri:%d, vi:%d] = %f\n", real_i, virt_i, C[n*col + real_i]);

        double koeff = C[n*col + real_i] / max_elem;
        // if j == col => update L
        if(j == col){
            C[n*j + real_i] = koeff;
        }else{ // else update U
            C[n*j + real_i] -= koeff * C[n*j + p[col]];
        }
    }
}




int main(){
    unsigned n;
    cin >> n;
    // alloc mem to union matrix(see wiki algorithm)
    host_vector<double> h_C(n * n);
    device_vector<double> d_C(n * n);
    host_vector<unsigned> h_ansvec(n);
    host_vector<unsigned> h_p(n);
    device_vector<unsigned> d_p(n);

    // input of matrix
    for(unsigned i = 0; i < n; ++i){
        h_ansvec[i] = i; // init of permutation vector
        for(unsigned j = 0; j < n; ++j){
            cin >> h_C[j*n + i]; // we store need matrix in  transpose format here for easy thrust search
        }
    }

    // transporting mem to device:
    d_p = h_ansvec;
    d_C = h_C;

    // pointers to mem:
    double* raw_C = thrust::raw_pointer_cast(d_C.data());
    unsigned* raw_p = thrust::raw_pointer_cast(d_p.data());

    // compute  LU
    try{
        for(unsigned i = 0; i < n - 1; ++i){
            // search index of max elem in col
            auto it_beg = make_permutation_iterator(d_C.begin() + i*n, d_p.begin());
            auto it_end = make_permutation_iterator(d_C.begin() + i*n, d_p.end());

            auto max_elem = max_element(it_beg + i, it_end);
            unsigned max_idx = max_elem - it_beg;
            double max_val = *max_elem;

            //swap(d_p[i], d_p[max_idx])
            {
                unsigned temp = d_p[i];
                d_p[i] = d_p[max_idx];
                d_p[max_idx] = temp;
            }

            h_ansvec[i] = max_idx;
            cout << "Max idx:" << max_idx << " max val:" << *max_val << endl;

            gauss_step<<<BLOCKS, THREADS>>>(raw_C, raw_p, n, i, *max_val);
            throw_on_cuda_error(cudaGetLastError(), i);

            throw_on_cuda_error(cudaThreadSynchronize(), i);
        }
    }catch(runtime_error& err){
        cout << "ERROR: " << err.what() << endl;
    }

    h_C = d_C;
    h_p = d_p;

    // get true order for output:
    map<unsigned, unsigned> order;
    for(unsigned i = 0; i < n; ++i){
        order[h_p[i]] = i;
    }

    // output for matrix:
    for(unsigned i = 0; i < n; ++i){
        unsigned row_num = order[i];
        for(unsigned j = 0; j < n; ++j){
            if(j){
                cout << " ";
            }

            cout << h_C[j*n + row_num];
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