// Made by Max Bronnikov
#ifndef CUDA_IMAGE_H
#define CUDA_IMAGE_H

#include <iostream>
#include <iomanip>
#include <string.h>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>

using namespace std;

// max threads is 512 in block => sqrt(512) is dim
#define MAX_X 16
#define MAX_Y 16
#define BLOCKS_X 32
#define BLOCKS_Y 32

#define RED(x) (x)&255
#define GREEN(x) ((x) >> 8)&255
#define BLUE(x) ((x) >> 16)&255
#define ALPHA(x) ((x) >> 24)&255


#define MAX_CLASS_NUMBERS 32

#define __RELEASE__
// #define __TIME_COUNT__
#define __WITH_IMG__

#define GREY(x) 0.299*((float)((x)&255)) + 0.587*((float)(((x)>>8)&255)) + 0.114*((float)(((x)>>16)&255))


// 2 dimentional texture
texture<uint32_t, 2, cudaReadModeElementType> g_text;


// filter(variant #8)
__global__ void sobel(uint32_t* d_data, uint32_t h, uint32_t w){
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t idy = blockIdx.y * blockDim.y + threadIdx.y;

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
    ans ^= (gradf << 16);
    ans ^= (gradf << 8);
    ans ^= (gradf);

    // locate in global mem
    d_data[idy*w + idx] = ans;
}


struct class_data{
    float avg_red;
    float avg_green;
    float avg_blue;

    float cov11;
    float cov12;
    float cov13;

    float cov21;
    float cov22;
    float cov23;

    float cov31;
    float cov32;
    float cov33;

    float log_cov;
};


__constant__ class_data computation_data[MAX_CLASS_NUMBERS];


// classsificator(variant #1)
__global__ void classification(uint32_t* picture, uint32_t h, uint32_t w, uint8_t classes){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t step_x = blockDim.x * gridDim.x;
    uint32_t step_y = blockDim.y * gridDim.y;

    // run for axis y
    for(uint32_t i = idy; i < h; i += step_y){
        // run for axis x
        for(uint32_t j = idx; j < w; j += step_x){
            // init very big num
            float min = (float) INT64_MAX;
            uint8_t ans_c = 0;

            uint32_t pixel = picture[i*w + j];
            /*
            ans_c = ALPHA(pixel);
            // if alpha  is exist => not need to compute
            if(ans_c){
                continue;
            }
            */

            for(uint8_t c = 0; c < classes; ++c){
                float red = RED(pixel);
                float green = GREEN(pixel);
                float blue = BLUE(pixel);

                float metric = 0.0;

                red -= computation_data[c].avg_red;
                green -= computation_data[c].avg_green;
                blue -= computation_data[c].avg_blue;

                float temp_red = red*computation_data[c].cov11 + 
                    green*computation_data[c].cov21 + blue*computation_data[c].cov31;

                float temp_green = red*computation_data[c].cov12 + 
                    green*computation_data[c].cov22 + blue*computation_data[c].cov32;
                
                float temp_blue = red*computation_data[c].cov13 + 
                    green*computation_data[c].cov23 + blue*computation_data[c].cov33;
                
                // dot + log(|cov|)
                metric = temp_red*red + temp_green*green + temp_blue*blue + computation_data[c].log_cov;
                
                if(metric < min){
                    ans_c = c;
                    min = metric;
                }

		if(idy == 100 && idx == 100){
                    printf("[%d, %d](%d) = %f\n", i, j, c, metric);
            	}
            }

           
            
            if(idy == 100 && idx == 100){
                printf("[%d, %d]/[%d, %d] step:%d\n", i, j, h, w, step_y);
            }
            
            // set pixel alpha chanel

            #ifndef __WITH_IMG__
            pixel ^= ((uint32_t) ans_c) << 24;
            #endif
            
            #ifdef __WITH_IMG__
            uint32_t color1 = (uint32_t) computation_data[ans_c].avg_red;
            uint32_t color2 = (uint32_t) computation_data[ans_c].avg_green;
            uint32_t color3 = (uint32_t) computation_data[ans_c].avg_blue;

            pixel = 0;
            pixel ^= color1;
            pixel ^= color2 << 8;
            pixel ^= color3 << 16;
            #endif

            picture[i*w + j] = pixel;
        }
    }
}



// exceptions if error
void throw_on_cuda_error(const cudaError_t& code)
{
  if(code != cudaSuccess)
  {
    throw std::runtime_error(cudaGetErrorString(code));
  }
}




// Image
class CUDAImage{
public:
    CUDAImage() : _data(nullptr), _widht(0), _height(0){

    }

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
                    uint32_t alpha = 0;
                    is.read(reinterpret_cast<char*>(&img._data[i + img._height*j]), sizeof(uint32_t));
                    alpha = ALPHA(img._data[i + img._height*j]);
                    img._data[i + img._height*j] ^= alpha << 24;
                }
            }
            std::swap(img._widht, img._height);
        }else{
            for(uint32_t i = 0; i < img._height; ++i){
                for(uint32_t j = 0; j < img._widht; ++j){
                    uint32_t alpha = 0;
                    is.read(reinterpret_cast<char*>(&img._data[i*img._widht + j]), sizeof(uint32_t));
                    alpha = ALPHA(img._data[i*img._widht + j]);
                    img._data[i*img._widht + j] ^= alpha << 24;
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


        #ifdef __TIME_COUNT__
        cudaEvent_t start, stop;
        float gpu_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        #endif

	    sobel<<<blocks, threads>>>(d_data, _height, _widht);
	    throw_on_cuda_error(cudaGetLastError());


        #ifdef __TIME_COUNT__
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);

        // open log:
        ofstream log("logs.log", ios::app);
        // title
        log << "GPU threads: " << MAX_X * MAX_Y << endl;
        // size:
        log << _height << " " << _widht << endl;
        // time:
        log << gpu_time << endl;
        log.close();
        #endif
        // run filter
        

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


    void cuda_classify_pixels(const vector<vector<uint32_t>>& indexes){
        class_data cov_avg[MAX_CLASS_NUMBERS];

        // memset(cov_avg, 0, sizeof(class_data));

        // compute data for classification
        compute_conv_avg(cov_avg, indexes);

        uint32_t* d_data = nullptr;

        throw_on_cuda_error(
            cudaMalloc((void**)&d_data, sizeof(uint32_t) * _widht * _height)
        );

        throw_on_cuda_error(
            cudaMemcpy(d_data, _data, sizeof(uint32_t) * _widht * _height, cudaMemcpyHostToDevice)
        );

        throw_on_cuda_error(
            cudaMemcpyToSymbol(computation_data, cov_avg, 
                MAX_CLASS_NUMBERS*sizeof(class_data), 0, cudaMemcpyHostToDevice)
        );

        dim3 threads = dim3(MAX_X, MAX_Y);
        dim3 blocks = dim3(BLOCKS_X, BLOCKS_Y);

        #ifdef __TIME_COUNT__
        cudaEvent_t start, stop;
        float gpu_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        #endif

        classification<<<blocks, threads>>>(d_data, _height, _widht, indexes.size());
        throw_on_cuda_error(cudaGetLastError());

        #ifdef __TIME_COUNT__
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);
        // open log:
        ofstream log("logs.log", ios::app);
        // title
        log  << BLOCKS_X * BLOCKS_Y * MAX_X * MAX_Y << endl;
        // size:
        log << _widht * _height << endl;
        // time:
        log << gpu_time << endl;
        log.close();
        #endif

        throw_on_cuda_error(
            cudaMemcpy(
                _data, d_data,
                sizeof(uint32_t) * _widht * _height,
                cudaMemcpyDeviceToHost
            )
        ); 

        throw_on_cuda_error(cudaFree(d_data));
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

    void compute_conv_avg(class_data* cov_avg, const vector<vector<uint32_t>>& indexes){
        // for all classes
        for(uint32_t i = 0; i < indexes.size(); ++i){
            double avg_red = 0.0;
            double avg_green = 0.0;
            double avg_blue = 0.0;

            double cov[9];
            memset(cov, 0, sizeof(double)*9);
            
            // npj
            uint32_t size = indexes[i].size() >> 1;

            // compute  avg
            for(uint32_t j = 0; j < indexes[i].size(); j += 2){
                uint32_t pixel = 0;
                // read pixel and update alpha if exist
                if(_transpose){
                    pixel = _data[indexes[i][j]*_widht + indexes[i][j+1]];
                    uint8_t alpha = i;
                    pixel ^= ((uint32_t)alpha) << 24;
                    _data[indexes[i][j]*_widht + indexes[i][j+1]] = pixel;
                }else{
                    pixel = _data[indexes[i][j+1]*_widht + indexes[i][j]];
                    uint8_t alpha = i;
                    pixel ^= ((uint32_t)alpha) << 24;
                    _data[indexes[i][j+1]*_widht + indexes[i][j]] = pixel;
                }
                avg_red += (double) (RED(pixel)); 
                avg_green += (double) (GREEN(pixel));
                avg_blue += (double) (BLUE(pixel)); 
            }
            
            avg_red /= size;
            avg_green /= size;
            avg_blue /= size;


            // write avg
            cov_avg[i].avg_red = (float) avg_red;
            cov_avg[i].avg_green = (float) avg_green;
            cov_avg[i].avg_blue = (float) avg_blue;


            // compute cov
            for(uint32_t j = 0; j < indexes[i].size(); j+=2){
                uint32_t pixel = 0;
                if(_transpose){
                    pixel = _data[indexes[i][j]*_widht + indexes[i][j+1]];
                }else{
                    pixel = _data[indexes[i][j+1]*_widht + indexes[i][j]];
                }
                double first = (double) (RED(pixel));
                first -= avg_red; 
                double second = (double) (GREEN(pixel));
                second -= avg_green;
                double third = (double) (BLUE(pixel));
                third -= avg_blue; 

                cov[0] += first*first; // 11
                cov[1] += first*second; // 12
                cov[2] += first*third; // 13

                cov[3] += first*second; // 21
                cov[4] += second*second; // 22
                cov[5] += second*third; // 23

                cov[6] += third*first; // 31 
                cov[7] += third*second; // 32
                cov[8] += third*third; // 33
            }

            cov[0] /= size - 1;
            cov[1] /= size - 1;
            cov[2] /= size - 1;

            cov[3] /= size - 1;
            cov[4] /= size - 1;
            cov[5] /= size - 1;

            cov[6] /= size - 1;
            cov[7] /= size - 1;
            cov[8] /= size - 1;

            // compute back:
            back_matrix(cov);

            // write back:
            cov_avg[i].cov11 = (float) cov[0];
            cov_avg[i].cov12 = (float) cov[1];
            cov_avg[i].cov13 = (float) cov[2];

            cov_avg[i].cov21 = (float) cov[3];
            cov_avg[i].cov22 = (float) cov[4];
            cov_avg[i].cov23 = (float) cov[5];

            cov_avg[i].cov31 = (float) cov[6];
            cov_avg[i].cov32 = (float) cov[7];
            cov_avg[i].cov33 = (float) cov[8];


            // compute log modulo:
            cov_avg[i].log_cov = log_of_modulo(cov);
        }
    }

    static float log_of_modulo(double* matr){
        double ans = 0.0;
        for(int i = 0; i < 9; ++i){
            ans += matr[i] * matr[i];
        }
        // if  |cov|  == 0 => log is wery small number
        return  ans > 0 ? (float) log(sqrt(ans)) : (float) INT64_MAX;
    }

    static void back_matrix(double* matr){
        double A11 = matr[4]*matr[8] - matr[5]*matr[7];
        double A12 = matr[5]*matr[6] - matr[3]*matr[8];
        double A13 = matr[3]*matr[7] - matr[4]*matr[6];
 
        double A21 = matr[2]*matr[7] - matr[1]*matr[8];
        double A22 = matr[0]*matr[8] - matr[2]*matr[6];
        double A23 = matr[1]*matr[6] - matr[0]*matr[7];

        double A31 = matr[1]*matr[5] - matr[2]*matr[4];
        double A32 = matr[2]*matr[3] - matr[0]*matr[5];
        double A33 = matr[0]*matr[4] - matr[1]*matr[3];

        double D = A11 * matr[0] + A12 * matr[1] + A13 * matr[2];

        matr[0] = A11 / D;
        matr[1] = A21 / D;
        matr[2] = A31 / D;
        
        matr[3] = A12 / D;
        matr[4] = A22 / D;
        matr[5] = A32 / D;

        matr[6] = A13 / D;
        matr[7] = A23 / D;
        matr[8] = A33 / D;
    }

    uint32_t* _data;
    uint32_t _height;
    uint32_t _widht;
    uint8_t _transpose;
};


#endif
