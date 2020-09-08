#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "cuda_image.h"

using namespace std;

int main(){
    string path1, path2;
    CUDAImage img;
    //cin >> path1 >> path2;
    //ifstream fin(path1);
    //ofstream fout(path2);
    cin >> img;
    try{
        img.FilterImg();
        cout << img;
    }catch(std::runtime_error &exception){
        cout << "ERROR: " << exception.what() << endl;
        return 0;
    }

    //fout.close();
    //fin.close();
    return 0;
}
