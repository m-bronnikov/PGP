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
#define MAX_X 3
#define MAX_Y 3

#define RED(x) (x)&255
#define GREEN(x) ((x) >> 8)&255
#define BLUE(x) ((x) >> 16)&255

#define __DEBUG__

#define GREY(x) 0.299*((float)((x)&255)) + 0.587*((float)(((x)>>8)&255)) + 0.114*((float)(((x)>>16)&255))


// 2 dimentional texture
texture<uint32_t, 2, cudaReadModeElementType> g_text;

// filter(variant #8)
__global__ void sobel(uint32_t* d_data, uint32_t h, uint32_t w){
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    printf("x = [%d %d]\n", blockIdx.x, threadIdx.x);
    printf("y = [%d %d]\n", blockIdx.y, threadIdx.y);

    if(idx >= w || idy >= h){
        return;
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
    int32_t gradf = (int32_t)sqrt(Gx*Gx + Gy*Gy);
    // max(grad, 255)
    
    gradf = gradf > 255 ? 255 : gradf;
    // store values in variable for minimize work with global mem
    ans ^= (gradf << 24);
    ans ^= (gradf << 16);
    ans ^= (gradf << 8);

    // locate in global mem
    d_data[idy*w + idx] = ans;
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
        ifstream fin(path, ios::in | ios::binary);
        if(!fin.is_open()){
            cout << "ERROR" << endl;
            throw std::runtime_error("cant open file!");
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


        #ifndef __RELEASE__
        uint32_t temp;
        os.unsetf(ios::dec);
        os.setf(ios::hex);
        if(img._transpose){
            temp = CUDAImage::reverse(img._height);
            os << setfill('0') << setw(8) <<  temp  << " ";
            temp = CUDAImage::reverse(img._widht);
            os << setfill('0') << setw(8) <<  temp  << endl;
        }else{
            temp = CUDAImage::reverse(img._widht);
            os << setfill('0') << setw(8) <<  temp  << " ";
            temp = CUDAImage::reverse(img._height);
            os << setfill('0') << setw(8) <<  temp  << endl;
        }
        #endif

        #ifdef __RELEASE__
        if(img._transpose){
            os.write(reinterpret_cast<const char*>(&img._height), sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(&img._widht), sizeof(uint32_t));
        }else{
            os.write(reinterpret_cast<const char*>(&img._widht), sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(&img._height), sizeof(uint32_t));
        }
        #endif

        #ifndef __RELEASE__
        if(img._transpose){
            for(uint32_t i = 0; i < img._widht; ++i){
                for(uint32_t j = 0; j < img._height; ++j){
                    if(j){
                        os << " ";
                    }
                    os << setfill('0') << setw(8) << reverse(img._data[j*img._widht + i]);
                }
                os << endl;
            }
        }else{
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    if(j){
                        os << " ";
                    }
                    os << setfill('0') << setw(8) << reverse(img._data[i*img._widht + j]);
                }
                os << endl;
            }
        }
        #endif
        
        #ifdef __RELEASE__
        if(img._transpose){
            for(uint32_t i = 0; i < img._widht; ++i){
                for(uint32_t j = 0; j < img._height; ++j){
                    os.write(reinterpret_cast<const char*>(&img._data[j*img._widht + i]), sizeof(uint32_t));
                }
            }
        }else{
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    os.write(reinterpret_cast<const char*>(&img._data[i*img._widht + j]), sizeof(uint32_t));
                }
            }
        }   
        #endif     


        #ifndef __RELEASE__

        os.unsetf(ios::hex);
        os.setf(ios::dec);

        #endif

        return os;
    }


    friend istream& operator>>(istream& is, CUDAImage& img){
        #ifndef __RELEASE__
        is.unsetf(ios::dec);
        is.setf(ios::hex);

        uint32_t temp;
        is >> temp;
        img._widht = CUDAImage::reverse(temp);
        is >> temp;
        img._height = CUDAImage::reverse(temp);

        #endif

        #ifdef __RELEASE__
        is.read(reinterpret_cast<char*>(&img._widht), sizeof(uint32_t));
        is.read(reinterpret_cast<char*>(&img._height), sizeof(uint32_t));
        #endif

        img._data = (uint32_t*) realloc(img._data, sizeof(uint32_t)*img._widht*img._height);
        img._transpose = img._widht >= img._height ? 0 : 1;

        #ifndef __RELEASE__
        if(img._transpose){
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    is >> img._data[i + img._height*j];
                    img._data[i + img._height*j] = reverse(img._data[i + img._height*j]);
                }
            }
            std::swap(img._widht, img._height);
        }else{
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    is >> img._data[i*img._widht + j];
                    img._data[j + img._widht*i] = reverse(img._data[j + img._widht*i]);
                }
            }
        }

        is.unsetf(ios::hex);
        is.setf(ios::dec);

        #endif

        #ifdef __RELEASE__

        if(img._transpose){
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    is.read(reinterpret_cast<char*>(&img._data[i + img._height*j]), sizeof(uint32_t));
                }
            }
            std::swap(img._widht, img._height);
        }else{
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    is.read(reinterpret_cast<char*>(&img._data[i*img._widht + j]), sizeof(uint32_t));
                }
            }
        }

        #endif 
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

        g_text.addressMode[0] = cudaAddressModeClamp;
        g_text.addressMode[1] = cudaAddressModeClamp;
        // cout << cudaAddressModeClamp << cudaAddressModeWrap << endl;


        uint32_t bloks_x = _height / MAX_X;
        uint32_t bloks_y = _widht / MAX_Y;

        bloks_x += bloks_x * MAX_X < _height ? 1 : 0;
        bloks_y += bloks_y * MAX_Y < _widht ? 1 : 0;

        dim3 threads = dim3(MAX_X, MAX_Y);
        dim3 blocks = dim3(bloks_y, bloks_x);

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
