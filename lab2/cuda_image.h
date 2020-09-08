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
    if(idx > h || idy > w){
        return;
    }
    printf("[%d, %d, %d] = %d\n", idx, idy, 0, tex3D(g_text, idx, idy, 0));
}


class CUDAImage{
public:
    CUDAImage() : _canals(3), _data(nullptr), _width(0), _height(0){}

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
        _height = _width = 0;
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
        img._height = CUDAImage::reverse(tmp);
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
        //uint8_t* h_data = (uint8_t*) malloc(3*sizeof(uint8_t)*_widht*_height);
        uint8_t* d_data = nullptr;
        cudaArray* a_data = nullptr;

        g_text.addressMode[0] = cudaAddressModeClamp;
        g_text.addressMode[1] = cudaAddressModeClamp;
        g_text.addressMode[2] = cudaAddressModeClamp;
        g_text.normilized = false;

        cudaChannelFormatDesc cfDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatUnsigned);

        CUDA_SAFE_CALL(cudaMallocArray(&a_data, &cfDesc, _canals * _widht, _height));
        CUDA_SAFE_CALL(cudaMemcpyToArray(
                                         a_data, 0, 0, _data,
                                         sizeof(uint8_t) * _canals * _widht * _height,
                                         cudaMemcpyHostToDevice
                                        ));

        CUDA_SAFE_CALL(cudaBindTextureToArray(g_text, a_data));

        CUDA_SAFE_CALL(cudaMalloc((void**)&d_data,
                                  sizeof(uint8_t) * _widht * _height * _canals));

        uint32_t bloks_x = _height / MAX_X;
        uint32_t bloks_y = _widht / MAX_Y;

        bloks_x += bloks_x * MAX_X < _height ? 1 : 0;
        bloks_y += bloks_y * MAX_Y < _widht ? 1 : 0;

        dim3 threads = dim3(MAX_X, MAX_Y, _canals);
        dim3 blocks = dim3(bloks_x, bloks_y);

        sobel<<<blocks, threads>>>(d_data, uint32_t _widht, uint32_t _height);

        CUDA_SAFE_CALL(cudaMemcpy(
                                  _data, d_data,
                                  sizeof(uint8_t) * _widht * _height * _canals,
                                  cudaMemcpyDeviceToHost
                                  ));

        CUDA_SAFE_CALL(cudaUnbindTexture(g_text));
        CUDA_SAFE_CALL(cudaFree(d_data));
        CuDA_SAFE_CALL(cudaFreeArray(a_data));
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
