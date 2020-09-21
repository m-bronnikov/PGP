#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include "cuda_image.h"

using namespace std;


int main(){
    string path1, path2;
    CUDAImage img;
    uint32_t size = 0;

    cin >> path1 >> path2;
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

    ifstream fin(path1, ios::in | ios::binary);
    ofstream fout(path2, ios::out | ios::binary);

    if(!fin || !fout){
        cout << "ERROR: " << "Cant open file" << endl;
        fin.close();
        fout.close();
        return 0;
    }

    fin >> img;

    try{
        img.cuda_classify_pixels(points);
        fout << img;
    }catch(std::runtime_error &exception){
        cout << "ERROR: " << exception.what() << endl;
    }

    fout.close();
    fin.close();
    return 0;
}
