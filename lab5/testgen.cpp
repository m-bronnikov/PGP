#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <time.h>

using namespace std;

#define INT_LIMIT 16777216 // 2^24

const string separate_line = "===============================";
const string done = " - DONE";
const string fail = " - FAIL";

int main(int argc, char const *argv[])
{
    srand(time(0));

    if(argc != 3){
        cerr << "Enter filename(1) and testsize(2) please! Exit!" << endl;
        return 1;
    }

    const char* path = argv[1];

    int size = atoi(argv[2]);
    vector<uint32_t> test(size);

    cout << separate_line << endl;
    cout << "GENERATE TEST";



    for(int i = 0; i < size; ++i){
        test[i] = rand() % INT_LIMIT;
    }

    cout << done << endl;
    cout << separate_line << endl;


    cout << endl;
    cout << separate_line << endl;
    cout << "OPEN FILE";

    ofstream file(path, ios::out);

    if(not file){
        cout << fail << endl;
        cout << separate_line << endl;
        return 1;
    }

    cout << done << endl;
    cout << separate_line << endl;

    cout << endl;
    cout << separate_line << endl;
    cout << "WRITE TEST";


    file.write(reinterpret_cast<const char*>(&size), sizeof(int));
    file.write(reinterpret_cast<const char*>(test.data()), size*sizeof(int));

    cout << done << endl;
    cout << separate_line << endl;


    cout << endl;
    cout << separate_line << endl;
    cout << "CLOSE FILE";

    file.close();

    cout << done << endl;
    cout << separate_line << endl;

    return 0;
}
