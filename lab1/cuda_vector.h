// Made by Max Bronnikov 
#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H

#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace std;

#define  MAXDELTA 10000
#define MAXPTHS 32
#define BLOCKS 32 // 1048576

#define __DEBUG__

template<typename T>
__global__ void elem_min(T* d_left, T* d_right, int size){
    int idx = blockDim.x * blockIdx.x +  threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for(int i = idx; i < size; i += step){
        T l_v = d_left[i];
        T r_v = d_right[i];
        d_left[i] = l_v < r_v ? l_v : r_v;
    }
}

template<typename T>
class CUDAvector{
public:
    CUDAvector(){
        _data = nullptr;
        _size = 0;
        _capacity = 0;
    }

    CUDAvector(size_t s) : _size(s), _capacity(s){
        _data = (T*) malloc(sizeof(T) * _size);
    }

    ~CUDAvector(){
        free(_data);
        _size = _capacity = 0;
    }

    // out is not parallel because order is important
    friend ostream& operator<<(ostream& os, const CUDAvector<T>& vec){
        for(size_t i = 0; i < vec._size; ++i){
            if(i){
                os << " ";
            }
            os << std::scientific << std::setprecision(10) << vec[i];
        }
        return os;
    }

    const T& operator[](const size_t index) const{
        return _data[index];
    }

    T& operator[](const size_t index){
        return _data[index];
    }

    void resize(size_t s){
        if(_capacity < s){
            _capacity = s;
            _data = (T*) realloc(_data, sizeof(T)*_capacity);
        }
        _size = s;
    }

    void reserve(size_t c){
        _capacity = c;
        _data = (T*) realloc(_data, sizeof(T)*_capacity);
    }

    void clear(){
        free(_data);
        _size = _capacity = 0;
    }

    void push_back(const T& val){
        if(_capacity == _size){
            _capacity += MAXDELTA > (_size << 1) ? MAXDELTA : (_size << 1);
            _data = (T*) realloc(_data, sizeof(T) * _capacity); 
        }
        _data[_size++] = val;
    }

    size_t size(){
        return _size;
    }

    size_t capacity(){
        return _capacity;
    }

    
    // parallel function
    friend void min2(CUDAvector<T>& left, CUDAvector<T>& right, CUDAvector<T>& ans){
        if(left._size != right._size){
            throw "Size is not equal!";
        }

        ans.resize(left._size);

        // CUDA mem alloc
        T* d_left;
        T* d_right;
        //T* d_ans;

        cudaError_t err = cudaMalloc((void**) &d_left, sizeof(T) * left._size);
        if (err != cudaSuccess){
            printf("ERROR: %s\n", cudaGetErrorString(err));
            exit(0);
        }

        err = cudaMalloc((void**) &d_right, sizeof(T) * right._size);
        if (err != cudaSuccess){
            printf("ERROR: %s\n", cudaGetErrorString(err));
            exit(0);
        }
        //cudaMalloc((void**) &d_ans, sizeof(T) * ans._size);

        // cpy mem to device
        err = cudaMemcpy(d_left, left._data, sizeof(T) * left._size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            printf("ERROR: %s\n", cudaGetErrorString(err));
            exit(0);
        }

        err = cudaMemcpy(d_right, right._data, sizeof(T) * right._size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            printf("ERROR: %s\n", cudaGetErrorString(err));
            exit(0);
        }


        #ifdef __DEBUG__
        cudaEvent_t start, stop;
        float gpu_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        #endif

        elem_min<<<BLOCKS, MAXPTHS>>>(d_left, d_right, ans._size);
        err = cudaGetLastError();
        // check errors
        if (err != cudaSuccess){
            printf("ERROR: %s\n", cudaGetErrorString(err));
            exit(0);
        }

        #ifdef __DEBUG__
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);

        // open log:
        ofstream log("logs.log", ios::app);
        // title
        log << MAXPTHS * BLOCKS << endl;
        // size:
        log << ans._size << endl;
        // time:
        log << gpu_time << endl;
        log.close();
        #endif

        // get ans from devise
        err = cudaMemcpy(ans._data, d_left, sizeof(T) * ans._size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
            printf("ERROR: %s\n", cudaGetErrorString(err));
            exit(0);
        }

        // free mem
        err = cudaFree(d_left);
        if (err != cudaSuccess){
            printf("ERROR: %s\n", cudaGetErrorString(err));
            exit(0);
        }
        err = cudaFree(d_right);
        if (err != cudaSuccess){
            printf("ERROR: %s\n", cudaGetErrorString(err));
            exit(0);
        }
    }

private:

    T* _data;
    size_t _size;
    size_t _capacity;
};


#endif
