#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>


using namespace std;
using namespace std::chrono;

#define RED(x) (x)&255
#define GREEN(x) ((x) >> 8)&255
#define BLUE(x) ((x) >> 16)&255
#define ALPHA(x) ((x) >> 24)&255

#define MAX_CLASS_NUMBERS 32

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

    float log_det;
};

// constant memory
class_data computation_data[MAX_CLASS_NUMBERS];

void classify_cpu(vector<vector<uint32_t>>& picture, uint32_t h, uint32_t w, uint8_t classes){

    // run for axis y
    for(uint32_t i = 0; i < h; ++i){
        // run for axis x
        for(uint32_t j = 0; j < w; ++j){
            // init very big num
            float min =  INT32_MAX;

            uint32_t pixel = picture[i][j];
            uint8_t ans_c = 0;

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
                metric = (temp_red*red + temp_green*green + temp_blue*blue + computation_data[c].log_det);
                
                if(metric < min){
                    ans_c = c;
                    min = metric;
                }
            }

            pixel ^= ((uint32_t) ans_c) << 24;


            picture[i][j] = pixel;
        }
    }
}



using namespace std;


int main(){
    string path;
    uint32_t w, h;

    uint32_t size = 0;

    cin >> path;
    cin >> size;

    vector<vector<uint32_t>> points(size);

    for(uint32_t i = 0; i < points.size(); ++i){
        uint32_t nums;
        cin >> nums;
        for(uint32_t j = 0; j < nums; ++j){
            uint32_t n1;
            cin >> n1;
            points[i].push_back(n1);
            cin >> n1;
            points[i].push_back(n1);
        }
    }


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
    classify_cpu(img, h, w, size);
    auto end = steady_clock::now();
    fout << "CPU" << endl;
    fout << h << " " << w << endl;
    fout << ((double)duration_cast<microseconds>(end - start).count()) / 1000.0 << endl;
    fout.close();

    return 0;
}