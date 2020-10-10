#include <iostream> 

using namespace std;

int main(){
    int n;
    //cout << "Enter matrix size: ";
    cin >> n;
    cout << n << endl;
    for(int i = 0; i < n; ++i){
        for(int j = 0; j  < i; ++j){
            cout << "10.0 ";
        }
        cout << (double)i * 0.1 + 0.3 << " ";
        for(int j = i + 1; j < n; ++j){
            cout << "10.0 ";
        }
        cout << endl;
    }
    return 0;
}