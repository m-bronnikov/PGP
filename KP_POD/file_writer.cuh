// Made by Max Bronnikov
#ifndef __FILE_WRITER_CUH__
#define __FILE_WRITER_CUH__

#include <fstream>
#include <cstdio>
#include <stdexcept>
 
using namespace std;

// FileWriter - class which gets GPU global memory buffer and writer img to file
class FileWriter{
public:
    FileWriter(uint32_t w, uint32_t h, const string& modifier, bool is_gpu) 
    : path_mod(modifier), size(w * h), width(w), height(h), gpu_data(is_gpu){
        h_data = is_gpu ? (new uint32_t[size]) : nullptr;
        if(gpu_data && !h_data){
            throw runtime_error("Allocation Error");
        }
    }

    ~FileWriter(){
        if(gpu_data){
            delete[] h_data;
        }
    }

    void write_to_file(uint32_t* d_data, uint32_t file_num){
        sprintf(buff, path_mod.c_str(), file_num); // set file path
        ofstream ofile(buff, ios::binary | ios::out); //  open file
        if(!ofile){
            throw runtime_error("File open error!");
        }

        write_to_stream(d_data, ofile); // write to file
        ofile.close();
    }

private:
    // colors - pointer to gpu global mem
    void write_to_stream(uint32_t* d_data, ostream& os){
        // copy data to host
        if(gpu_data){
            throw_on_cuda_error(cudaMemcpy(h_data, d_data, size*sizeof(int32_t), cudaMemcpyDeviceToHost));
        }else{
            h_data = d_data;
        }

        // write data to stream
        os.write(reinterpret_cast<char*>(&width), sizeof(uint32_t));
        os.write(reinterpret_cast<char*>(&height), sizeof(uint32_t));
        os.write(reinterpret_cast<char*>(h_data), size * sizeof(uint32_t));
    }

private:
    bool gpu_data;

    string path_mod;
    char buff[256];
    uint32_t* h_data;
    //uint32_t* d_data;
    uint32_t size;
    uint32_t width;
    uint32_t height;
};




#endif // __FILE_WRITER_CUH__