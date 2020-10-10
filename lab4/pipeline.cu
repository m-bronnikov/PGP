// Made by Max Bronnikov
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <iomanip>

using namespace std;
using namespace thrust;

const unsigned BLOCKS = 1024;
const unsigned THREADS = 1024;

struct abs_functor : public thrust::unary_function<double, double>{
    __host__ __device__
    double operator()(double elem) const {
        return elem < 0.0 ? -elem : elem;
    }
};

struct abs_comparator{
    abs_functor fabs;

    __host__ __device__ double operator()(double a, double b){
        return fabs(a) < fabs(b);
    }
};


void throw_on_cuda_error(const cudaError_t& code){
    if(code != cudaSuccess){
        throw runtime_error(cudaGetErrorString(code));
    }
}

__global__ void gauss_step_L(double* C, unsigned n, unsigned col, double max_elem){
    unsigned thrd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned thrd_step = blockDim.x * gridDim.x;

    //unsigned start_pounsigned = (col + 1) >> 8;
    unsigned idx0 = n * col;

    for(unsigned index = thrd_idx + col + 1; index < n; index += thrd_step){
        C[idx0 + index] /= max_elem;
    }
}

__global__ void gauss_step_U(double* C, unsigned n, unsigned col){
    unsigned i_idx = threadIdx.x;
    unsigned j_idx = blockIdx.x;

    unsigned i_step = blockDim.x;
    unsigned j_step = gridDim.x;


    for(unsigned jndex = j_idx + col + 1; jndex < n; jndex += j_step){
        unsigned idx0 = jndex*n;
        double C_jc = C[idx0 + col];
    
        for(unsigned index = i_idx + col + 1; index < n; index += i_step){
            C[idx0 + index] -= C[n*col + index] * C_jc;
        }
    }
}

__global__ void swap_lines(double* C, unsigned n, unsigned line1, unsigned line2){
    unsigned thrd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned thrd_step = blockDim.x * gridDim.x;

    //unsigned start_pounsigned = (col + 1) >> 8;
    for(unsigned index = thrd_idx; index < n; index += thrd_step){
        double temp = C[index*n + line1];
        C[index*n + line1] = C[index*n + line2];
        C[index*n + line2] = temp;
    }
}


int main(){
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    unsigned n;
    cin >> n;

    // alloc mem to union matrix(see wiki algorithm)
    //int size = get_allign_size(n);

    double* h_C = (double*) malloc(n * n * sizeof(double));
    unsigned* h_p = (unsigned*) malloc(n * sizeof(unsigned));
    double* d_C;
    throw_on_cuda_error(cudaMalloc((void**) &d_C, sizeof(double) * n * n));

    // input of matrix
    for(unsigned i = 0; i < n; ++i){
        h_p[i] = i; // init of permutation vector
        for(unsigned j = 0; j < n; ++j){
            cin >> h_C[j*n + i]; // we store need matrix in  transpose format here for easy thrust search
        }
    }

    // transporting mem to device:
    //d_p = h_ansvec
    throw_on_cuda_error(cudaMemcpy(d_C, h_C, sizeof(double) * n * n, cudaMemcpyHostToDevice));

    // compute  LU
    try{
        for(unsigned i = 0; i < n - 1; ++i){
            // search index of max elem in col
            //auto it_beg = make_transform_iterator(d_C.begin() + i*size, abs_functor());

            auto it_beg = thrust::device_pointer_cast(d_C + i*n);

            auto max_elem = thrust::max_element(it_beg + i, it_beg + n, abs_comparator());

            unsigned max_idx = max_elem - it_beg;
            double max_val = *max_elem;

            if(i != max_idx){
                swap_lines<<<BLOCKS, THREADS>>>(d_C, n, i, max_idx);
                h_p[i] = max_idx;
                //throw_on_cuda_error(cudaGetLastError(), i);
                cudaThreadSynchronize();
            }

            gauss_step_L<<<BLOCKS, THREADS>>>(d_C, n, i, max_val);
            //throw_on_cuda_error(cudaGetLastError(), i);
            cudaThreadSynchronize();

            gauss_step_U<<<BLOCKS, THREADS>>>(d_C, n, i);
            throw_on_cuda_error(cudaGetLastError());
            //throw_on_cuda_error(cudaThreadSynchronize(), i);
        }
    }catch(runtime_error& err){
        cout << "ERROR: " << err.what() << endl;
    }

    throw_on_cuda_error(cudaMemcpy(h_C, d_C, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    throw_on_cuda_error(cudaFree(d_C));


    for(unsigned i = 0; i < n; ++i){
        for(unsigned j = 0; j < n; ++j){
            cout << std::scientific << std::setprecision(10) << h_C[j*n + i] << " ";
        }
        cout << endl;
    }
    // output of vector
    for(unsigned i = 0; i < n; ++i){
        cout << h_p[i] << " ";
    }
    cout << endl;

    free(h_C);
    free(h_p);

    return 0;
}
