#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "cuda_image.h"

using namespace std;


int main(){
    string path1, path2;
    CUDAImage img;

    cin >> path1 >> path2;

    ifstream fin(path1, ios::in | ios::binary);
    ofstream fout(path2, ios::out | ios::binary);

    fin >> img;

    try{
        img.cuda_filter_img();
    }catch(std::runtime_error &exception){
        cout << "ERROR: " << exception.what() << endl;
        //return 0;
    }

    fout << img;

    fout.close();
    fin.close();
    return 0;
}
