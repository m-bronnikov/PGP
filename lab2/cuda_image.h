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
#include <algorithm>

using namespace std;

// max threads is 512 in block => sqrt(512) is dim
#define MAX_X 22
#define MAX_Y 22

#define RED(x) ((x) >> 24)
#define GREEN(x) ((x) >> 16)&255
#define BLUE(x) ((x) >> 8)&255

#define GREY(x) 0.299*((float)((x)>>24)) + 0.587*((float)(((x)>>16)&255)) + 0.114*((float)(((x)>>8)&255))


// 2 dimentional texture
texture<uint32_t, 2, cudaReadModeElementType> g_text;

// filter(variant #8)
__global__ void sobel(uint32_t* d_data, uint32_t h, uint32_t w){
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx >= h || idy >= w){
        return;
    }

    {
        uint32_t w22 = tex2D(g_text, idx, idy);
        printf(
            "[%d, %d] = %d %d %d 00\n", idx, idy,
            RED(w22), GREEN(w22), BLUE(w22)
        );
    }

    // ans pixel
    uint32_t ans = 0;

    // locate area in mem(32 bite)
    // compute grey scale for all pixels in area
    float w11 = GREY(tex2D(g_text, idx - 1, idy - 1));
    float w12 = GREY(tex2D(g_text, idx, idy - 1));
    float w13 = GREY(tex2D(g_text, idx + 1, idy - 1));
    float w21 = GREY(tex2D(g_text, idx - 1, idy));

    float w23 = GREY(tex2D(g_text, idx + 1, idy));
    float w31 = GREY(tex2D(g_text, idx - 1, idy + 1));
    float w32 = GREY(tex2D(g_text, idx, idy + 1));
    float w33 = GREY(tex2D(g_text, idx + 1, idy + 1));

    // compute Gx Gy
    float Gx = w13 + w23 + w23 + w33 - w11 - w21 - w21 - w31;
    float Gy = w31 + w32 + w32 + w33 - w11 - w12 - w12 - w13;

    // full gradient
    uint32_t gradf = (uint32_t)sqrt(Gx*Gx + Gy*Gy);
    // max(grad, 255)
    gradf = gradf > 255 ? 255 : gradf;
    // store values in variable for minimize work with global mem
    ans ^= (gradf << 24);
    ans ^= (gradf << 16);
    ans ^= (gradf << 8);

    // locate in global mem
    d_data[idx*w + idy] = ans;
    /*
    // red:
    int32_t G1 = RED(w13) + (RED(w23) << 1) + RED(w33) - RED(w11) - (RED(w21) << 1) - RED(w31);
    int32_t G2 = RED(w31) + (RED(w32) << 1) + RED(w33) - RED(w11) - (RED(w12) << 1) - RED(w13);
    uint32_t gradf = sqrt((double)(G1*G1 + G2*G2));
    gradf = gradf > 255 ? 255 : gradf;
    ans ^= (gradf << 24);

    {
        uint32_t w22 = tex2D(g_text, idx, idy);
        printf(
            "RED[%d, %d] = %d, G = [%d, %d]\n", idx, idy,
            gradf, G1, G2
        );
    }

    // green:
    G1 = GREEN(w13) + (GREEN(w23) << 1) + GREEN(w33) - GREEN(w11) - (GREEN(w21) << 1) - GREEN(w31);
    G2 = GREEN(w31) + (GREEN(w32) << 1) + GREEN(w33) - GREEN(w11) - (GREEN(w12) << 1) - GREEN(w13);
    gradf = sqrt((double)(G1*G1 + G2*G2));
    gradf = gradf > 255 ? 255 : gradf;
    ans ^= (gradf << 16);

    // blue:
    G1 = BLUE(w13) + (BLUE(w23) << 1) + BLUE(w33) - BLUE(w11) - (BLUE(w21) << 1) - BLUE(w31);
    G2 = BLUE(w31) + (BLUE(w32) << 1) + BLUE(w33) - BLUE(w11) - (BLUE(w12) << 1) - BLUE(w13);
    gradf = sqrt((double)(G1*G1 + G2*G2));
    gradf = gradf > 255 ? 255 : gradf;
    ans ^= (gradf << 8);

    // to global mem
    d_data[idx*w + idy] = ans;
    */
}

// exceptions if error
void throw_on_cuda_error(cudaError_t code)
{
  if(code != cudaSuccess)
  {
    throw std::runtime_error(cudaGetErrorString(code));
  }
}


// Image
class CUDAImage{
public:
    CUDAImage() : _data(nullptr), _widht(0), _height(0){}

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

        if(img._transpose){
            temp = CUDAImage::reverse(img._height);
            cout << setfill('0') << setw(8) <<  temp  << " ";
            temp = CUDAImage::reverse(img._widht);
            cout << setfill('0') << setw(8) <<  temp  << endl;
        }else{
            temp = CUDAImage::reverse(img._widht);
            cout << setfill('0') << setw(8) <<  temp  << " ";
            temp = CUDAImage::reverse(img._height);
            cout << setfill('0') << setw(8) <<  temp  << endl;
        }

        if(img._transpose){
            for(uint32_t i = 0; i < img._widht; ++i){
                for(uint32_t j = 0; j < img._height; ++j){
                    if(j){
                        cout << " ";
                    }
                    cout << setfill('0') << setw(8) << img._data[j*img._widht + i];
                }
                cout << endl;
            }
        }else{
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    if(j){
                        cout << " ";
                    }
                    cout << setfill('0') << setw(8) << img._data[i*img._widht + j];
                }
                cout << endl;
            }
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
        img._data = (uint32_t*) realloc(img._data, sizeof(uint32_t)*img._widht*img._height);

        img._transpose = img._widht >= img._height ? 0 : 1;

        if(img._transpose){
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    cin >> img._data[i + img._height*j];
                }
            }
            std::swap(img._widht, img._height);
        }else{
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    cin >> img._data[i*img._widht + j];
                }
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

    // 10000
    // make filration
    void cuda_filter_img(){
        // device data
        uint32_t* d_data = nullptr;
        cudaArray* a_data = nullptr;

        // prepare data

        // out:
        throw_on_cuda_error(
            cudaMalloc((void**)&d_data, sizeof(uint32_t) * _widht * _height)
        );

        // texture:
        cudaChannelFormatDesc cfDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

        throw_on_cuda_error(
            cudaMalloc3DArray(&a_data, &cfDesc, {_widht, _height, 0})
        );

        throw_on_cuda_error(
            cudaMemcpy2DToArray(
                    a_data, 0, 0, _data, _widht * sizeof(uint32_t),
                    _widht * sizeof(uint32_t), _height, cudaMemcpyHostToDevice
            )
        );

        throw_on_cuda_error(
            cudaBindTextureToArray(g_text, a_data, cfDesc)
        );

        // use clamp optimisation for limit exits

        g_text.normalized = false;

        g_text.addressMode[0] = cudaAddressModeMirror;
        g_text.addressMode[1] = cudaAddressModeMirror;
        // cout << cudaAddressModeClamp << cudaAddressModeWrap << endl;


        uint32_t bloks_x = _height / MAX_X;
        uint32_t bloks_y = _widht / MAX_Y;

        bloks_x += bloks_x * MAX_X < _height ? 1 : 0;
        bloks_y += bloks_y * MAX_Y < _widht ? 1 : 0;

        dim3 threads = dim3(MAX_X, MAX_Y);
        dim3 blocks = dim3(bloks_x, bloks_y);

        // run filter
        sobel<<<blocks, threads>>>(d_data, _height, _widht);
        throw_on_cuda_error(cudaGetLastError());


        throw_on_cuda_error(
            cudaMemcpy(
                _data, d_data,
                sizeof(uint32_t) * _widht * _height,
                cudaMemcpyDeviceToHost
            )
        );

        throw_on_cuda_error(cudaUnbindTexture(g_text));
        throw_on_cuda_error(cudaFree(d_data));
        throw_on_cuda_error(cudaFreeArray(a_data));

    }


private:
    // MSB-first to LSB-first:
    static uint32_t reverse(uint32_t num){
        uint32_t ans = 0;
        for(uint32_t i = 0; i < 4; ++i){
            uint32_t temp = (num >> (24 - 8*i)) & 255;
            ans ^= (temp << 8*i);
        }
        return ans;
    }

    uint32_t* _data;
    uint32_t _height;
    uint32_t _widht;
    uint8_t _transpose;
};


#endif
