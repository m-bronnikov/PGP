#include <iostream>
#include <vector>
#include <stdlib.h>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;


int main(){
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    unsigned n;
    cin >> n;

    // alloc mem to union matrix(see wiki algorithm)
    //int size = get_allign_size(n);

    vector<double> h_C(n * n);
    vector<unsigned> h_p(n);

    // input of matrix
    for(unsigned i = 0; i < n; ++i){
        h_p[i] = i; // init of permutation vector
        for(unsigned j = 0; j < n; ++j){
            cin >> h_C[i*n + j]; // we store need matrix in  transpose format here for easy thrust search
        }
    }

    auto start = steady_clock::now();

    // compute  LU
    for(unsigned i = 0; i < n - 1; ++i){

        auto it_beg = begin(h_C) + i*n;

        unsigned max_idx = i;
        double max_val = h_C[i*n + i];

        for(int j = i + 1; j < n; ++j){
            if(std::fabs(h_C[j*n + i]) > std::fabs(max_val)){
                max_val = h_C[j*n + i];
                max_idx = j;
            }
        } 

        h_p[i] = max_idx;

        // swipe lines:
        for(int j = 0; j < n; ++j){
            std::swap(h_C[i*n + j], h_C[max_idx*n + j]);
        }

        // L step:
        for(int j = i  + 1; j < n; ++j){
            h_C[j * n + i] /= max_val;
        }

        for(int j = i + 1; j < n; ++j){
            for(int k = i + 1; k < n; ++k){
                h_C[j*n + k] -= h_C[i*n + k] * h_C[j*n + i];
            }
        }
    }

    auto end = steady_clock::now();

    ofstream fout("logs.log", ios::app);
    fout << "CPU" << endl;
    fout << 0 << endl;
    fout << n << endl;
    fout << ((double)duration_cast<microseconds>(end - start).count()) / 1000.0 << endl;
    fout.close();

    return 0;
}
