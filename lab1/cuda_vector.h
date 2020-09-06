// Made by Max Bronnikov 
#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H

#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define  MAXDELTA 10000

template<typename T>
__global__ void elem_min(T* d_left, T* d_right, T* d_ans){
    int idx = threadIdx.x;
    T l_v = d_left[idx];
    T r_v = d_right[idx];
    d_ans[idx] = l_v < r_v ? l_v : r_v;
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
            os << std::fixed << std::setprecision(10) << vec[i];
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
        T* d_ans;

        cudaMalloc((void**) &d_left, sizeof(T) * left._size);
        cudaMalloc((void**) &d_right, sizeof(T) * right._size);
        cudaMalloc((void**) &d_ans, sizeof(T) * ans._size);

        // cpy mem to device
        cudaMemcpy(d_left, left._data, sizeof(T) * left._size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_right, right._data, sizeof(T) * right._size, cudaMemcpyHostToDevice);

        // start ans.size kernels for parallel work on threads
        elem_min<<<1, ans._size>>>(d_left, d_right, d_ans);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
            exit(0);

        // get ans from devise
        cudaMemcpy(ans._data, d_ans, sizeof(T) * ans._size, cudaMemcpyDeviceToHost);
        // free mem

        cudaFree(d_left);
        cudaFree(d_right);
        cudaFree(d_ans);
    }

private:

    T* _data;
    size_t _size;
    size_t _capacity;
};


#endif