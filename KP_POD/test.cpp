#include <iostream>

using namespace std;

int func(int a = 8){
    return a;
}

int main(){
    int a = 5;

    cout << func() << endl;


    return 0;
}