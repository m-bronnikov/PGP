#include <iostream>
#include <string>

using namespace std;

int main(){
    int a = 5;

    auto f = [&a](int i){
        return a*i;
    };

    for(a = 0; a < 5; ++a){
        cout << f(1) << endl;
    }

    return 0;
}