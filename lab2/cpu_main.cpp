#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>


using namespace std;
using namespace std::chrono;


#define GREY(x) 0.299*((float)((x)&255)) + 0.587*((float)(((x)>>8)&255)) + 0.114*((float)(((x)>>16)&255))




uint32_t tex2D(const vector<vector<uint32_t>>& vec, int32_t i, int32_t j){
    i = i < vec.size() ? i : vec.size() - 1;
    i = i < 0 ? 0 : i;

    j = j < vec[0].size() ? j : vec[0].size() - 1;
    j = j < 0 ? 0 : j;

    return vec[i][j];
}

void sobel_cpu(const vector<vector<uint32_t>>& g_text, vector<vector<uint32_t>>& d_data){
    for(int i = 0; i < g_text.size(); ++i){
        for(int j = 0; j < g_text[i].size(); ++j){
            uint32_t ans = 0;

            float w11 = GREY(tex2D(g_text, i - 1, j - 1));
            float w12 = GREY(tex2D(g_text, i, j - 1));
            float w13 = GREY(tex2D(g_text, i + 1, j - 1));
            float w21 = GREY(tex2D(g_text, i - 1, j));

            float w23 = GREY(tex2D(g_text, i + 1, j));
            float w31 = GREY(tex2D(g_text, i - 1, j + 1));
            float w32 = GREY(tex2D(g_text, i, j + 1));
            float w33 = GREY(tex2D(g_text, i + 1, j + 1));

            // compute Gx Gy
            float Gx = w13 + w23 + w23 + w33 - w11 - w21 - w21 - w31;
            float Gy = w31 + w32 + w32 + w33 - w11 - w12 - w12 - w13;
            int32_t gradf = (int32_t)sqrt(Gx*Gx + Gy*Gy);
            // max(grad, 255)
    
            gradf = gradf > 255 ? 255 : gradf;
            // store values in variable for minimize work with global mem
            ans ^= (gradf << 16);
            ans ^= (gradf << 8);
            ans ^= (gradf);

            d_data[i][j] = ans;
        }
    }
}


int main(){
    string path;
    uint32_t w, h;
    cin >> path;
    ifstream fin(path);

    
    fin.read(reinterpret_cast<char*>(&w), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&h), sizeof(uint32_t));


    vector<vector<uint32_t>> img(h, vector<uint32_t>(w));
    vector<vector<uint32_t>> out(h, vector<uint32_t>(w));

    for(uint32_t i = 0; i < h; ++i){
        for(uint32_t j = 0; j < w; ++j){
            fin.read(reinterpret_cast<char*>(&img[i][j]), sizeof(uint32_t));
        }
    }
    ofstream fout("logs.log", ios::app);
    // timer:
    auto start = steady_clock::now();
    sobel_cpu(img, out);
    auto end = steady_clock::now();
    fout << "CPU" << endl;
    fout << h * w << endl;
    fout << ((double)duration_cast<microseconds>(end - start).count()) / 1000.0 << endl;
    fout.close();




    return 0;
}
