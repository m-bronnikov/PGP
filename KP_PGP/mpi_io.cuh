#ifndef __MPI_IO__
#define __MPI_IO__

#include "mpi.h"
#include <iostream>
#include <iomanip>

using namespace std;

// This class needs to read dta from union stream and send/recv to all procs
class MpiReader{
public:
    MpiReader(istream& to_read, int proc_num) : is(to_read), main_worker(proc_num == 0){}

    friend MpiReader& operator>>(MpiReader& talker, uint32_t& num){
        if(talker.main_worker){
            talker.is >> num;
        }

        MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);

        return talker;
    }

    friend MpiReader& operator>>(MpiReader& talker, float& num){
        if(talker.main_worker){
            talker.is >> num;
        }

        MPI_Bcast(&num, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        return talker;
    }

    friend MpiReader& operator>>(MpiReader& talker, string& str){
        int filename_size;

        if(talker.main_worker){
            talker.is >> str;
            filename_size = str.size();
        }

        MPI_Bcast(&filename_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        str.resize(filename_size);
        MPI_Bcast(const_cast<char*>(str.c_str()), filename_size, MPI_CHAR, 0, MPI_COMM_WORLD);

        return talker;
    }

private:
    bool main_worker;
    istream& is;
};

// This class needs to logging of frame rendering. It requires monotonus uniform work distribution.
class MpiLogger{
public:
    MpiLogger(ostream& to_write, int proc_rank, int procs_num)
    : os(to_write), num_of_procs(procs_num), id_of_proc(proc_rank) {
        if(id_of_proc == 0){
            os << left <<  setw(12) << "Frame";
            os << internal << setw(10) << "Render Time(ms)";
            os << right << setw(13) << "Max Rays" << endl;
        }
    }

    void write(int frame_num, int rays, int time){
        if(id_of_proc == 0){
            os << left <<  setw(12) << frame_num;
            os << internal << setw(10) << time / 1000.0f;
            os << right << setw(17) << rays << endl;

            // If some proces aborted => all processes after this also aborted.
            int min_count_procs = num_of_procs;
            for(int i = 1; i < num_of_procs; ++i){
                int alive;
                MPI_Status tmp;

                MPI_Recv(&alive, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &tmp);

                if(alive == 0){
                    min_count_procs = i < min_count_procs ? i : min_count_procs;
                    continue;
                }

                MPI_Recv(&frame_num, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &tmp);
                MPI_Recv(&rays, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &tmp);
                MPI_Recv(&time, 1, MPI_INT, i, 3, MPI_COMM_WORLD, &tmp);

                os << left << setw(12) << frame_num;
                os << internal << setw(10) << time / 1000.0f;
                os << right << setw(17) << rays << endl;
            }

            num_of_procs = min_count_procs;
        }else{
            // send status of living to main
            int alive = 1;
            MPI_Send(&alive, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            MPI_Send(&frame_num, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&rays, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&time, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        }
    }

    ~MpiLogger(){
        if(id_of_proc == 0){
            // If some proces aborted => all processes after this also aborted.
            for(int i = 1; i < num_of_procs; ++i){
                int alive;
                MPI_Status tmp;

                MPI_Recv(&alive, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &tmp);
            }
        }else{
            // send status of living to main
            int alive = 0;
            MPI_Send(&alive, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
private:
    ostream& os;
    int num_of_procs;
    int id_of_proc;
};

#endif