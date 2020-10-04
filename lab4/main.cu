// Made by Max Bronnikov
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

using namespace std;
using namespace thrust;

class CudaMatrix{

private:
    device_vector<unsigned> p;
    device_vector<double> matr;
};

int main(){
    unsigned n;
    cin >> n;
    // alloc mem to union matrix(see wiki algorithm)
    host_vector<double> h_C(n * n);
    host_vector<double> d_C(n * n);
    device_vector<unsigned> h_p(n);
    device_vector<unsigned> d_p(n);

    for(unsigned i = 0; i < n; ++i){
        h_p[i] = i; // init of permutation vector
        for(unsigned j = 0; j < n; ++j){
            cin >> h_C[j*n + i]; // we store need matrix in  transpose format here for easy thrust search
        }
    }

    // transpose mem to device:
    d_p = h_p;
    d_C = h_C;

    for
}