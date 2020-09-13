#include <iostream>
#include <fstream>
#include <random>
#include <string>

using namespace std;

int main(){
    int size;
    string path;
    cout << "Path to test: ";
    cin >> path;
    cout << "Size of vectors: ";
    cin >> size;

    ofstream fout(path);

    fout << size << endl;
    for(int i = 0; i < size; ++i){
        fout << (double)rand() / ((double)(rand() + 1));
    }
    fout << endl;
    for(int i = 0; i < size; ++i){
        fout << (double)rand() / ((double)(rand() + 1));
    }

    fout << endl;


    fout.close();
    return 0;
}