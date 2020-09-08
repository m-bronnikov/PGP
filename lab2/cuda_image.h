// Made by Max Bronnikov
#ifndef CUDA_IMAGE_H
#define CUDA_IMAGE_H

#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>

using namespace std;


#define MAX_X 13
#define MAX_Y 13

texture<uint8_t, 3, cudaReadModeElementType> g_text;

__global__ void sobel(uint8_t* ans, uint32_t w, uint32_t h){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idz = threadIdx.z;
    if(idx > h || idy > w){
        return;
    }
    printf("[%d, %d, %d] = %d\n", idx, idy, idz, tex3D(g_text, idx, idy, idz));
}

void throw_on_cuda_error(cudaError_t code)
{
  if(code != cudaSuccess)
  {
    throw std::runtime_error(cudaGetErrorString(code));
  }
}



class CUDAImage{
public:
    CUDAImage() : _canals(3), _data(nullptr), _widht(0), _height(0){}

    CUDAImage(const string& path) : CUDAImage(){
        ifstream fin(path);
        if(!fin.is_open()){
            cout << "ERROR" << endl;
            exit(0);
        }

        fin >> (*this);

        fin.close();
    }

    ~CUDAImage(){
        free(_data);
        _height = _widht = 0;
    }

    // out is not parallel because order is important
    friend ostream& operator<<(ostream& os, const CUDAImage& img){
        os.unsetf(ios::dec);
        os.setf(ios::hex | ios::uppercase);
        uint32_t temp;

        temp = CUDAImage::reverse(img._widht);
        cout << setfill('0') << setw(8) <<  temp  << " ";
        temp = CUDAImage::reverse(img._height);
        cout << setfill('0') << setw(8) <<  temp  << endl;

        for(uint32_t i = 0; i < img._height; ++i){
            for(uint32_t j = 0; j < img._widht; ++j){
                if(j){
                    cout << " ";
                }
                cout << setfill('0') << setw(2);
                cout << (uint32_t) img._data[3*i*img._widht + 3*j];

                cout << setfill('0') << setw(2);
                cout << (uint32_t) img._data[3*i*img._widht + 3*j + 1];

                cout << setfill('0') << setw(2);
                cout << (uint32_t) img._data[3*i*img._widht + 3*j + 2];

                cout << "00";
            }
            cout << endl;
        }

        os.unsetf(ios::hex);
        os.setf(ios::dec);
        return os;
    }


    friend istream& operator>>(istream& is, CUDAImage& img){
        is.unsetf(ios::dec);
        is.setf(ios::hex);

        uint32_t temp;
        cin >> temp;
        img._widht = CUDAImage::reverse(temp);
        cin >> temp;
        img._height = CUDAImage::reverse(temp);
        img._data = (uint8_t*) realloc(img._data, 3*sizeof(uint8_t)*img._widht*img._height);

        for(uint32_t i = 0; i < img._height; ++i){
            for(uint32_t j = 0; j < img._widht; ++j){
                cin >> temp;
                img._data[3*i*img._widht + 3*j] = (temp >> 24) & 255;
                img._data[3*i*img._widht + 3*j + 1] = (temp >> 16) & 255;
                img._data[3*i*img._widht + 3*j + 2] = (temp >> 8) & 255;
            }
        }

        is.unsetf(ios::hex);
        is.setf(ios::dec);
        return is;
    }


    void clear(){
        free(_data);
        _widht = _height = 0;
    }

    void FilterImg(){
        uint8_t* d_data = nullptr;
        cudaArray* a_data = nullptr;
        g_text.addressMode[0] = cudaAddressModeClamp;
        g_text.addressMode[1] = cudaAddressModeClamp;
        g_text.addressMode[2] = cudaAddressModeClamp;
        g_text.normalized = false;

        cudaChannelFormatDesc cfDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

        throw_on_cuda_error(
            cudaMallocArray(&a_data, &cfDesc, _canals * _widht * sizeof(uint8_t), _height)
        );

        throw_on_cuda_error(
            cudaMemcpy2DToArray(
                            a_data, 0, 0, _data, _canals * _widht * sizeof(uint8_t),
                            _canals * _widht * sizeof(uint8_t), _height, cudaMemcpyHostToDevice
            )
        );

        throw_on_cuda_error(
            cudaBindTextureToArray(g_text, a_data)
        );

        throw_on_cuda_error(
            cudaMalloc((void**)&d_data,
                    sizeof(uint8_t) * _widht * _height * _canals)
        );

        uint32_t bloks_x = _height / MAX_X;
        uint32_t bloks_y = _widht / MAX_Y;

        bloks_x += bloks_x * MAX_X < _height ? 1 : 0;
        bloks_y += bloks_y * MAX_Y < _widht ? 1 : 0;

        dim3 threads = dim3(MAX_X, MAX_Y, _canals);
        dim3 blocks = dim3(bloks_x, bloks_y);

        sobel<<<blocks, threads>>>(d_data, _widht, _height);
        throw_on_cuda_error(cudaGetLastError());

        //cout << "Here" << endl;
        throw_on_cuda_error(
            cudaMemcpy(
                _data, d_data,
                sizeof(uint8_t) * _widht * _height * _canals,
                cudaMemcpyDeviceToHost
            )
        );

        throw_on_cuda_error(cudaUnbindTexture(g_text));
        throw_on_cuda_error(cudaFree(d_data));
        throw_on_cuda_error(cudaFreeArray(a_data));
        
    }


private:
    static uint32_t reverse(uint32_t num){
        uint32_t ans = 0;
        for(uint32_t i = 0; i < 4; ++i){
            uint32_t temp = (num >> (24 - 8*i)) & 255;
            ans ^= (temp << 8*i);
        }
        return ans;
    }

    uint8_t* _data;
    size_t _height;
    size_t _widht;
    const size_t _canals;
};


#endif
