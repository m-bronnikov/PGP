#include <iostream>
#include <iomanip>
#include <fstream>


using namespace std;

int main(){
    uint8_t b;
    cin.unsetf(ios::dec);
    cin.setf(ios::hex);
    cin >> b;
    cout.unsetf(ios::dec);
    cout.setf(ios::hex | ios::uppercase);
    cout << setfill('0') << setw(2) <<  (uint32_t)b << endl;
    return 0;
}