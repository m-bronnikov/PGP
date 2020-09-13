#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;


void elem_min_cpu(const vector<double>& in1, 
			const vector<double>& in2,
			vector<double>& out){
  for(size_t i = 0; i < in1.size(); ++i){
     out[i] = in1[i] < in2[i] ? in1[i] : in2[i];
  }
}

int main(){
  int s;
  cin >> s;
  vector<double> in1(s);
  vector<double> in2(s);
  vector<double> ans(s);

  for(auto& elem : in1){
    cin >> elem;
  }

  for(auto& elem : in2){
    cin >> elem;
  }
  // for logs:
  ofstream fout("logs.log", ios::app);
  // timer:
  auto start = steady_clock::now();
  elem_min_cpu(in1, in2, ans);
  auto end = steady_clock::now();
  fout << "CPU" << endl;
  fout << s << endl;
  fout << ((double)duration_cast<microseconds>(end - start).count()) / 1000.0 << endl;
  fout.close();
  return 0;
}
