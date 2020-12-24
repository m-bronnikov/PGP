// Made by Max Bronnikov
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <string.h>
#include <algorithm>

using namespace std;

#define INT_LIMIT 16777216 // 2^24
#define STD_CHECK false

const string separate_line = "===============================";
const string done = " - DONE";
const string fail = " - FAIL";


// count sort for faster check of large data
void cpu_count_sort(int32_t* data, uint32_t size){
    if(!size){
        return;
    }

    uint32_t* counter = new uint32_t[INT_LIMIT];
    memset(counter, 0, INT_LIMIT*sizeof(uint32_t));

    // histogram
    for(uint32_t j = 0; j < size; j++){
        ++counter[data[j]];
    }

    // exclusive scan
    uint32_t sum = 0;
    for(uint32_t j = 0; j < INT_LIMIT; ++j){
        uint32_t temp = counter[j];
        counter[j] = sum;
        sum += temp;
    }

    // Sorting:

    // without last element
    for(uint32_t j = 0; j < INT_LIMIT - 1; ++j){
        while(counter[j] < counter[j + 1]){
            data[counter[j]++] = j;
        }
    }

    // last element
    {
        while(counter[INT_LIMIT - 1] < size){
            data[counter[INT_LIMIT - 1]++] = INT_LIMIT - 1;
        }  
    }

    delete[] counter;
}


int main(int argc, char const *argv[])
{
    if(argc != 3){
        cerr << "Enter source filename(1) and resulting filename(2) please! Exit!" << endl;
        return 1;
    }

    cout << separate_line << endl;
    cout << "OPEN FILES";

    // open files
    ifstream source(argv[1], ios::binary | ios::in);
    ifstream sorted(argv[2], ios::binary | ios::in);

    if(not source || not sorted){
        cout << fail;
        cout << separate_line << endl;
        return 1;
    }

    cout << done << endl;
    cout << separate_line << endl;

    cout << endl;
    cout << separate_line << endl;
    cout << "READ DATA";

    // get size from source
    uint32_t size = 0;
    source.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));

    // alloc data
    vector<int32_t> source_seq(size), sorted_seq(size);

    // read data
    sorted.read(reinterpret_cast<char*>(&sorted_seq[0]), size*sizeof(int32_t));
    source.read(reinterpret_cast<char*>(&source_seq[0]), size*sizeof(int32_t));

    cout << done << endl;
    cout << separate_line << endl;

    cout << endl;
    cout << separate_line << endl;
    cout << "CLOSE FILES";
    source.close();
    sorted.close();

    cout << done << endl;
    cout << separate_line << endl;

    cout << endl;
    cout << separate_line << endl;
    cout << "START COMPARE DATA" << endl;
    cout << separate_line << endl;


    if(STD_CHECK){
        sort(begin(source_seq), end(source_seq)); // this implementation 100% correct but O(n logn)
    }else{
        cpu_count_sort(reinterpret_cast<int32_t*>(source_seq.data()), size); // this implementation faster O(n)
    }

    // compare seqs
    string status = "SUCCESS";
    for(int i = 0; i < size; ++i){
        if(sorted_seq[i] != source_seq[i]){
            cout << "ERROR!" << endl;
            cout << "Position №" << i << " | Expected value: " << source_seq[i] << " | Occured value: " << sorted_seq[i] << endl;
            cout << separate_line << endl;
            status = "FAIL";
            break;
        }
    }

    cout << "STATUS: " << status << endl;
    cout << separate_line << endl;
    cout << "END COMAPARE DATA" << endl;
    cout << separate_line << endl;

    return 0;
}