#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <iomanip>
#include <string.h>
#include "mpi.h"

#define RELEASE

using namespace std;

//#define ndims 3
//#define ndims_x_2 6

const int ndims = 3;
const int ndims_x_2 = 6;

/*CODE UPDATING:*/

int is_main(int worker){
    return worker ? 0 : 1;
}

double u_next(double ux0, double ux1, double uy0, double uy1, 
                        double uz0, double uz1, double h2x, 
                        double h2y, double h2z){
    double ans = (ux0 + ux1) * h2x;
    ans += (uy0 + uy1) * h2y; 
    ans += (uz0 + uz1) * h2z;
    return ans;
}

double max_determine(double val1, double val2, double curr_max){
    double diff = val1 - val2;
    diff = diff < 0.0 ? -diff : diff;

    return diff > curr_max ? diff : curr_max;
}

void recv_waiting(MPI_Request* in, MPI_Request* out){
    MPI_Status temp;
    MPI_Wait(in, &temp);
    MPI_Wait(out, &temp);
}

void print_line(ostream& os, double* line, int size){
    for(int i = 0; i < size; ++i){
        os << line[i] << " ";
    }
}

// cpy memory from edge buffers to inside idges
void import_edges(double* inner_buff, double** edge_buffs, int* edge_sizes, int* blocks){
    int size_x = blocks[0] + 2;
    int size_y = blocks[1] + 2;
    int size_z = blocks[2] + 2;

    int start[ndims];
    int mult[ndims];
    int lims[ndims];
    int ones[ndims];

    lims[0] = ones[0] = size_x;
    lims[1] = ones[1] = size_y;
    lims[2] = ones[2] = size_z;
    --lims[0]; --lims[1]; --lims[2];

    mult[0] = mult[1] = mult[2] = 1;
    start[0] = start[1] = start[2] = 1;

    for(int orr = 0; orr < ndims_x_2; ++orr){
        int dir = orr >> 1;
        start[dir] = (blocks[dir] + 1) * (orr & 1);
        lims[dir] = (orr & 1) ? (blocks[dir] + 2) : 1;
        ones[dir] = 1;
        mult[dir] = 0;
 
        for(int k = start[2]; k < lims[2]; ++k){
            for(int j = start[1]; j < lims[1]; ++j){
                for(int i = start[0]; i < lims[0]; ++i){
                    inner_buff[i + (j + k*size_y)*size_x] = 
                        edge_buffs[orr][i*mult[0] + (j*mult[1] + k*mult[2]*ones[1])*ones[0]];
                }
            }
        }

        ones[dir] = lims[dir]--;
        start[dir] = mult[dir] = 1;
    }
}


// cpy memory from  inside edges to exchange buffers
void export_edges(double* inner_buff, double** edge_buffs, int* edge_sizes, int* blocks){
    int size_x = blocks[0] + 2;
    int size_y = blocks[1] + 2;
    int size_z = blocks[2] + 2;

    int start[ndims];
    int mult[ndims];
    int lims[ndims];
    int ones[ndims];

    lims[0] = ones[0] = size_x;
    lims[1] = ones[1] = size_y;
    lims[2] = ones[2] = size_z;
    --lims[0]; --lims[1]; --lims[2];

    mult[0] = mult[1] = mult[2] = 1;
    start[0] = start[1] = start[2] = 1;

    for(int orr = 0; orr < ndims_x_2; ++orr){
        int dir = orr >> 1;
        if(orr & 1){
            start[dir] = blocks[dir];
        }

        lims[dir] = (orr & 1) ? blocks[dir] + 1 : 2;
        
        ones[dir] = 1;
        mult[dir] = 0;
 
        for(int k = start[2]; k < lims[2]; ++k){
            for(int j = start[1]; j < lims[1]; ++j){
                for(int i = start[0]; i < lims[0]; ++i){ 
                    edge_buffs[orr][i*mult[0] + (j*mult[1] + k*mult[2]*ones[1])*ones[0]] = 
                        inner_buff[i + (j + k*size_y)*size_x];
                }
            }
        }

        ones[dir] = lims[dir] + 1;
        start[dir] = mult[dir] = 1;
    }
}


void edges_exchange(double** edge_buff_in, double** edge_buff_out, int* dir_edge_sizes, 
        int* coords, int* dimens, int* neighb_ranks, MPI_Comm grid_comm){

    MPI_Request in[ndims_x_2], out[ndims_x_2]; // statuses of exchange beetwen processes
    // send data
    for(int dir = 0; dir < ndims; ++dir){
        int dir_x_2 = dir << 1;
        if(coords[dir]){
            MPI_Isend(
                edge_buff_out[dir_x_2], 
                dir_edge_sizes[dir],
                MPI_DOUBLE,
                neighb_ranks[dir_x_2],
                0,
                grid_comm,
                &out[dir_x_2]
            );
            MPI_Irecv(
                edge_buff_in[dir_x_2], 
                dir_edge_sizes[dir],
                MPI_DOUBLE,
                neighb_ranks[dir_x_2],
                0,
                grid_comm,
                &in[dir_x_2]  
            );
        }
        if(coords[dir] < dimens[dir] - 1){
            MPI_Isend(
                edge_buff_out[dir_x_2 + 1], 
                dir_edge_sizes[dir],
                MPI_DOUBLE,
                neighb_ranks[dir_x_2 + 1],
                0,
                grid_comm,
                &out[dir_x_2 + 1]
            );
            MPI_Irecv(
                edge_buff_in[dir_x_2 + 1], 
                dir_edge_sizes[dir],
                MPI_DOUBLE,
                neighb_ranks[dir_x_2 + 1],
                0,
                grid_comm,
                &in[dir_x_2 + 1]
            );
        }
    }

    // wait data
    for(int dir = 0; dir < ndims; ++dir){
        int orr = dir << 1;
        if(coords[dir] > 0){
            recv_waiting(&in[orr], &out[orr]);
        }
        if(coords[dir] < dimens[dir] - 1){
            recv_waiting(&in[orr + 1], &out[orr + 1]);
        }
    }
}



int main(int argc, char **argv){
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    /*       DETERMINE ALGORITHM VARIABLES:      */
    int main_worker, proc_rank;
    int workers_count = 0;
    // input data
    int dimens[ndims], blocks[ndims];
    double l[ndims];
    double u[ndims_x_2];
    double u0, eps;
    string path;
    // num of each orientation and direction
    enum orientation{
        left = 0, right,
        front, back,
        down, up,
    };

    enum direction{
        dir_x = 0,
        dir_y,
        dir_z
    };

    // start of mpi usage
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    // init info about role of current worker
    main_worker = is_main(proc_rank);


    // work with stdin stream if main role  of worker
    if(main_worker){
        // read data by defclared in task order
        cin >> dimens[dir_x] >> dimens[dir_y] >> dimens[dir_z];
        cin >> blocks[dir_x] >> blocks[dir_y] >> blocks[dir_z];
        cin >> path;
        cin >> eps;
        cin >> l[dir_x] >> l[dir_y] >> l[dir_z];
        cin >> u[down] >> u[up];
        cin >> u[left] >> u[right];
        cin >> u[front] >> u[back];
        cin >> u0;
    }

    #ifdef TIME_COUNT
    double time_start;
    if(main_worker){
        time_start = MPI_Wtime();
    }
    #endif

    // send/recv all data
    MPI_Bcast(dimens, ndims, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(blocks, ndims, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l, ndims, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u, ndims_x_2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    /*    CREATE NEW ABSTRACT TOPOLOGY OF PARALLELEPIPED   */
    MPI_Comm grid_comm; // communicator
    int pereod[ndims], coords[ndims]; // coords of procs
    int neighb_ranks[ndims_x_2];

    fill_n(pereod, ndims, 0); // we have non-pereodical topology


    // init topology and place of current worker in it
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dimens, pereod, false, &grid_comm);
    MPI_Comm_rank(grid_comm, &proc_rank); // get rank in new topology
    MPI_Comm_size(grid_comm, &workers_count); // get num of all workers
    MPI_Cart_coords(grid_comm, proc_rank, ndims, coords); // get coords of worker
    // its main? (not needed and may be renmove)
    main_worker = is_main(proc_rank);

    // get ranks of left, right, up, down, front, back process:
    MPI_Cart_shift(grid_comm, dir_x, 1, &neighb_ranks[left], &neighb_ranks[right]); // left and right
    MPI_Cart_shift(grid_comm, dir_y, 1, &neighb_ranks[front], &neighb_ranks[back]); // front and back
    MPI_Cart_shift(grid_comm, dir_z, 1, &neighb_ranks[down], &neighb_ranks[up]); // down and up


    /* COMPUTATION VARIABLES: */

    // compute steps:
    double h2x, h2y, h2z; 
    h2x = l[dir_x] / ((double)dimens[dir_x]*blocks[dir_x]);
    h2y = l[dir_y] / ((double)dimens[dir_y]*blocks[dir_y]);
    h2z = l[dir_z] / ((double)dimens[dir_z]*blocks[dir_z]);

    h2x *= h2x;
    h2y *= h2y;
    h2z *= h2z;

    {
        double denuminator = 2.0*(1.0/h2x + 1.0/h2y + 1.0/h2z);
        h2x = 1.0 / (denuminator * h2x);
        h2y = 1.0 / (denuminator * h2y);
        h2z = 1.0 / (denuminator * h2z);
    }

 
    // compute sizes of edges for all directions
    int dir_edge_sizes[ndims];
    int size_x = blocks[dir_x] + 2;
    int size_y = blocks[dir_y] + 2;
    int size_z = blocks[dir_z] + 2;

    dir_edge_sizes[dir_x] = size_y * size_z;
    dir_edge_sizes[dir_y] = size_x * size_z;
    dir_edge_sizes[dir_z] = size_y * size_x;
    
    /* HEAP MEMORY ALLOC AND INIT */
    double* norm_data = new double[workers_count];
    double* edge_buff_out[ndims_x_2]; 
    double* edge_buff_in[ndims_x_2];
    double* buffer0;
    double* buffer1;
    // init and alloc main buffes
    {
        int buff_size = size_x * size_y * size_z;
        buffer0 = new double[buff_size];
        buffer1 = new double[buff_size];
        fill_n(buffer0, buff_size, u0);
    }

    // alloc and init memory for transport  buffers
    for(int dir = 0; dir < ndims; ++dir){
        // alloc
        int dir_x_2 = dir << 1;
        edge_buff_out[dir_x_2] = new double[dir_edge_sizes[dir]];
        edge_buff_in[dir_x_2] = new double[dir_edge_sizes[dir]];
        edge_buff_out[dir_x_2 + 1] = new double[dir_edge_sizes[dir]];
        edge_buff_in[dir_x_2 + 1] = new double[dir_edge_sizes[dir]];

        // init :
        fill_n(edge_buff_out[dir_x_2], dir_edge_sizes[dir], u0);
        fill_n(edge_buff_out[dir_x_2 + 1], dir_edge_sizes[dir], u0);

        // if edge proc => set buffers with given edge values
        if(!coords[dir]){
            fill_n(edge_buff_in[dir_x_2], dir_edge_sizes[dir], u[dir_x_2]);
        }
        if(coords[dir] == dimens[dir] - 1){
            fill_n(edge_buff_in[dir_x_2 + 1], dir_edge_sizes[dir], u[dir_x_2 + 1]);
        }
    }

    /* INDEXING IN BUFFERS: */
    auto idx = [size_x, size_y](int i, int j, int k){
        return (i + size_x*(j + k*size_y));
    };

    /* START OF ITTERATIVE COMPUTING: */

    double max_diff = 0.0; // maximum error

    do{
        max_diff = 0.0;
        // Step 1(Send and recv all data):
        edges_exchange(edge_buff_in, edge_buff_out, dir_edge_sizes, 
            coords, dimens, neighb_ranks, grid_comm);

        // Step 2(Fill in edges)
        import_edges(buffer0, edge_buff_in, dir_edge_sizes, blocks);

        // Step 3(compute new values):
        for(int k = 1; k <= blocks[dir_z]; ++k){
            for(int j = 1; j <= blocks[dir_y]; ++j){
                for(int i = 1; i <= blocks[dir_x]; ++i){
                    // new value:
                    buffer1[idx(i, j, k)] = u_next(
                        buffer0[idx(i - 1, j, k)], buffer0[idx(i + 1, j, k)],
                        buffer0[idx(i, j - 1, k)], buffer0[idx(i, j + 1, k)],
                        buffer0[idx(i, j, k - 1)], buffer0[idx(i, j, k + 1)],
                        h2x, h2y, h2z
                    );
                    // compute max abs(diff):
                    max_diff = max_determine(buffer0[idx(i, j, k)], buffer1[idx(i, j, k)], max_diff);
                }
            }
        }

        // Step 4(Fill out edges)
        export_edges(buffer1, edge_buff_out, dir_edge_sizes, blocks);

        // Step 5(Control of stopping):
        MPI_Allgather(&max_diff, 1, MPI_DOUBLE, norm_data, 1, MPI_DOUBLE, grid_comm);
        max_diff = 0.0;
        for(int i = 0; i < workers_count; ++i){
            max_diff = max_diff < norm_data[i] ? norm_data[i] : max_diff; // maximum
        }


        // Step 6(swap buffers):
        double* tmp = buffer1;
        buffer1 = buffer0;
        buffer0 = tmp;
    }while(max_diff >= eps);

    // output of data:
    ofstream fout;
    if(main_worker){
        fout.open(path, ios::out);
        fout << std::scientific << std::setprecision(7);
    }

    int temp_coord[ndims];
    MPI_Status status;

    /* OUTPUT */
    // we will send by one buffer, then call Barier on workers
    // amin worker will recv them and print already
    for(int k = 0; k < dimens[dir_z]; ++k){
        temp_coord[dir_z] = k;
        for(int kk = 1; kk <= blocks[dir_z]; ++kk){
            for(int j = 0; j < dimens[dir_y]; ++j){
                temp_coord[dir_y] = j;
                for(int jj = 1; jj <= blocks[dir_y]; ++jj){
                    for(int i = 0; i < dimens[dir_x]; ++i){
                        temp_coord[dir_x] = i;
                        // if main worker => recv data form friend
                        if(main_worker){
                            // if buffer from main => dont recv, just a print
                            if(coords[dir_z] == k && coords[dir_y] == j && coords[dir_x] == i){
                                print_line(fout, &buffer0[idx(1, jj, kk)], blocks[dir_x]);
                            // if not from worker, recv data and print
                            }else{
                                int rank = 0; 
                                MPI_Cart_rank(grid_comm, temp_coord, &rank);
                                // sync read
                                MPI_Recv(
                                    buffer1, blocks[dir_x], 
                                    MPI_DOUBLE, rank,
                                    0, grid_comm, &status
                                );
                                // when data stored, we can print it:
                                print_line(fout, buffer1, blocks[dir_x]);
                            }
                        // if it not main worker send data when our queue
                        }else{
                            if(coords[dir_z] == k && coords[dir_y] == j && coords[dir_x] == i){
                                // sync send
                                MPI_Send(
                                    &buffer0[idx(1, jj, kk)], blocks[dir_x],
                                    MPI_DOUBLE, 0, 0,
                                    grid_comm
                                ); // Bsend may be better
                            }
                        }
                    }
                }
            }
        }
    }

    // close Mpi connection
    MPI_Finalize();

    // free all memory
    delete[] norm_data;
    delete[] buffer0;
    delete[] buffer1;

    for(int dir = 0; dir < ndims; ++dir){
        int dir_x_2 = dir << 1;
        delete[] edge_buff_out[dir_x_2];
        delete[] edge_buff_in[dir_x_2];
        delete[] edge_buff_out[dir_x_2 + 1];
        delete[] edge_buff_in[dir_x_2 + 1];
    }    

    // close file if opened
    if(main_worker){
        fout << endl;
        fout.close();
    }

    #ifdef TIME_COUNT
    double time_end;
    if(main_worker){
        time_end = MPI_Wtime();
        cout << "Inference time: ";
        cout << (time_end - time_start) << "sec" << endl;
    }
    #endif

    return 0;
}