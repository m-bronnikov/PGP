// Made by Max Bronnikov
#include <iostream>
#include "cuda_vector.h"

using namespace std;

int main(){
    int n;
    cin >> n;

    CUDAvector<double> left(n), right(n);
    CUDAvector<double> ans;

    for(size_t i = 0; i < n; ++i){
        cin >> left[i];
    }

    for(size_t i = 0; i < n; ++i){
        cin >> right[i];
    }

    cout << left << endl;
    cout << right << endl;

    // main func (4 variant)
    min2(left, right, ans);

    cout << ans << endl;

    return 0;
}