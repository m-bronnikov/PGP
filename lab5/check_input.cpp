#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

#define INT_LIMIT 16777216 // 2^24

const string separate_line = "===============================";
const string done = " - DONE";
const string fail = " - FAIL";

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

    // sort source sequence
    sort(begin(source_seq), end(source_seq));

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