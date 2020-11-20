#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <iomanip>
#include <string.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include "mpi.h"

#define RELEASE

using namespace std;

//#define ndims 3
//#define ndims_x_2 6

const int ndims = 3;
const int ndims_x_2 = 6;

const int LITTLE_BLOCKS = 16;
const int LITTLE_THREADS = 16;

const int BLOCKS = 32;
const int THREADS = 32;

const int precission = 6;
const int float_size = 14; // 12 digits in one + separator + possible minus

/* CUDA FUNCTIONS */

#define next_ijk(i, j, k, step) { \
    i += step;  \
    while(i > n_x){ \
        i -= n_x; \
        ++j; \
    } \
    while(j > n_y){ \
        j -= n_y; \
        ++k; \
    } \
} \

#define next_ij(i, j, step) { \
    i += step;  \
    while(i > n_x){ \
        i -= n_x; \
        ++j; \
    } \
} \

#define next_ik(i, k, step) { \
    i += step;  \
    while(i > n_x){ \
        i -= n_x; \
        ++k; \
    } \
} \

#define next_jk(j, k, step) { \
    j += step;  \
    while(j > n_y){ \
        j -= n_y; \
        ++k; \
    } \
} \

#define idx(i, j, k) ((i) + size_x*((j) + (k)*size_y))


__host__ __device__ 
double u_next(double ux0, double ux1, double uy0, double uy1, 
                        double uz0, double uz1, double h2x, 
                        double h2y, double h2z){
    double ans = (ux0 + ux1) * h2x;
    ans += (uy0 + uy1) * h2y; 
    ans += (uz0 + uz1) * h2z;
    return ans;
}

__host__ __device__
double max_determine(double val1, double val2, double curr_max){
    double diff = val1 - val2;
    diff = diff < 0.0 ? -diff : diff;

    return diff > curr_max ? diff : curr_max;
}

__global__
void import_throught_x(double* inner_buff, double* edge_buff1, 
                                double* edge_buff2, int n_x, int n_y, int n_z){
    int size_x = n_x + 2;
    int size_y = n_y + 2;

    int num_threads = blockDim.x * gridDim.x;
    int thread_idx = blockDim.x * blockIdx.x +  threadIdx.x;

    int j = 1;
    int k = 1;

    next_jk(j, k, thread_idx);

    while(k <= n_z){
        inner_buff[idx(0, j, k)] = edge_buff1[j + k * size_y];
        inner_buff[idx(n_x + 1, j, k)] = edge_buff2[j + k * size_y];
        next_jk(j, k, num_threads);
    }
}

__global__
void import_throught_y(double* inner_buff, double* edge_buff1, 
                                double* edge_buff2, int n_x, int n_y, int n_z){
    int size_x = n_x + 2;
    int size_y = n_y + 2;

    int num_threads = blockDim.x * gridDim.x;
    int thread_idx = blockDim.x * blockIdx.x +  threadIdx.x;

    int i = 1;
    int k = 1;

    next_ik(i, k, thread_idx);

    while(k <= n_z){
        inner_buff[idx(i, 0, k)] = edge_buff1[i + k * size_x];
        inner_buff[idx(i, n_y + 1, k)] = edge_buff2[i + k * size_x];
        next_ik(i, k, num_threads);
    }
}

__global__
void import_throught_z(double* inner_buff, double* edge_buff1, 
                                    double* edge_buff2, int n_x, int n_y, int n_z){
    int size_x = n_x + 2;
    int size_y = n_y + 2;

    int num_threads = blockDim.x * gridDim.x;
    int thread_idx = blockDim.x * blockIdx.x +  threadIdx.x;

    int i = 1;
    int j = 1;

    next_ij(i, j, thread_idx);

    while(j <= n_y){
        inner_buff[idx(i, j, 0)] = edge_buff1[i + j * size_x];
        inner_buff[idx(i, j, n_z + 1)] = edge_buff2[i + j * size_x];
        next_ij(i, j, num_threads);
    }
}

__global__
void export_throught_x(double* inner_buff, double* edge_buff1, 
    double* edge_buff2, int n_x, int n_y, int n_z){
    int size_x = n_x + 2;
    int size_y = n_y + 2;

    int num_threads = blockDim.x * gridDim.x;
    int thread_idx = blockDim.x * blockIdx.x +  threadIdx.x;

    int j = 1;
    int k = 1;

    next_jk(j, k, thread_idx);

    while(k <= n_z){
        edge_buff1[j + k * size_y] = inner_buff[idx(1, j, k)];
        edge_buff2[j + k * size_y] = inner_buff[idx(n_x, j, k)];
        next_jk(j, k, num_threads);
    }
}

__global__
void export_throught_y(double* inner_buff, double* edge_buff1, 
    double* edge_buff2, int n_x, int n_y, int n_z){
    int size_x = n_x + 2;
    int size_y = n_y + 2;

    int num_threads = blockDim.x * gridDim.x;
    int thread_idx = blockDim.x * blockIdx.x +  threadIdx.x;

    int i = 1;
    int k = 1;

    next_ik(i, k, thread_idx);

    while(k <= n_z){
        edge_buff1[i + k * size_x] = inner_buff[idx(i, 1, k)];
        edge_buff2[i + k * size_x] = inner_buff[idx(i, n_y, k)];
        next_ik(i, k, num_threads);
    }
}

__global__
void export_throught_z(double* inner_buff, double* edge_buff1, 
        double* edge_buff2, int n_x, int n_y, int n_z){
    int size_x = n_x + 2;
    int size_y = n_y + 2;

    int num_threads = blockDim.x * gridDim.x;
    int thread_idx = blockDim.x * blockIdx.x +  threadIdx.x;

    int i = 1;
    int j = 1;

    next_ij(i, j, thread_idx);

    while(j <= n_y){
        edge_buff1[i + j * size_x] = inner_buff[idx(i, j, 1)];
        edge_buff2[i + j * size_x] = inner_buff[idx(i, j, n_z)];
        next_ij(i, j, num_threads);
    }
}

// compute grid
__global__
void compute_new_grid(double* buffer1, double* buffer0, double* max_values, 
    int n_x, int n_y, int n_z, double h2x, double h2y, double h2z
){
    int size_x = n_x + 2;
    int size_y = n_y + 2;

    int num_threads = blockDim.x * gridDim.x;
    int thread_idx = blockDim.x * blockIdx.x +  threadIdx.x;

    int i = 1;
    int j = 1;
    int k = 1;
    next_ijk(i, j, k, thread_idx);

    // variable for less global memory requests
    double temp;

    // currently max value is zero
    max_values[thread_idx] = 0.0;

    while(k <= n_z){
        // new value:
        temp = u_next(
            buffer0[idx(i - 1, j, k)], buffer0[idx(i + 1, j, k)],
            buffer0[idx(i, j - 1, k)], buffer0[idx(i, j + 1, k)],
            buffer0[idx(i, j, k - 1)], buffer0[idx(i, j, k + 1)],
            h2x, h2y, h2z
        );

        // compute max abs(diff):
        max_values[thread_idx] = max_determine(buffer0[idx(i, j, k)], temp, max_values[thread_idx]);
        buffer1[idx(i, j, k)] = temp;

        next_ijk(i, j, k, num_threads);
    }
}



/* MPI FUNCTIONS */
int is_main(int worker){
    return worker ? 0 : 1;
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
    int filename_size;
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
        filename_size = path.size();
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
    // filename sending:
    {
        MPI_Bcast(&filename_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        path.resize(filename_size);
        MPI_Bcast((char*) path.c_str(), filename_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

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

    /* SET CUDA DEVICE */
    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(proc_rank % device_count);


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
    double* h_edge_buff_out[ndims_x_2]; 
    double* h_edge_buff_in[ndims_x_2];
    double* h_buffer0;
    // double* h_buffer1;

    double* d_edge_buff_out[ndims_x_2]; 
    double* d_edge_buff_in[ndims_x_2];
    double* d_buffer0;
    double* d_buffer1;
    double* d_maxvalues;

    // init and alloc main buffes
    
    // host alloc 
    int buff_size = size_x * size_y * size_z;
    h_buffer0 = new double[buff_size];
    fill_n(h_buffer0, buff_size, u0);
    // cuda alloc
    cudaMalloc((void**) &d_buffer0, sizeof(double) * buff_size);
    cudaMalloc((void**) &d_buffer1, sizeof(double) * buff_size);
    cudaMemcpy(d_buffer0, h_buffer0, sizeof(double) * buff_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_maxvalues, sizeof(double) * BLOCKS*THREADS);
    

    // alloc and init memory for transport  buffers
    for(int dir = 0; dir < ndims; ++dir){
        // host alloc
        int dir_x_2 = dir << 1;
        h_edge_buff_out[dir_x_2] = new double[dir_edge_sizes[dir]];
        h_edge_buff_in[dir_x_2] = new double[dir_edge_sizes[dir]];
        h_edge_buff_out[dir_x_2 + 1] = new double[dir_edge_sizes[dir]];
        h_edge_buff_in[dir_x_2 + 1] = new double[dir_edge_sizes[dir]];

        // cuda alloc
        cudaMalloc((void**) &d_edge_buff_in[dir_x_2], sizeof(double) * dir_edge_sizes[dir]);
        cudaMalloc((void**) &d_edge_buff_in[dir_x_2 + 1], sizeof(double) * dir_edge_sizes[dir]);
        cudaMalloc((void**) &d_edge_buff_out[dir_x_2], sizeof(double) * dir_edge_sizes[dir]);
        cudaMalloc((void**) &d_edge_buff_out[dir_x_2 + 1], sizeof(double) * dir_edge_sizes[dir]);


        // init :
        fill_n(h_edge_buff_out[dir_x_2], dir_edge_sizes[dir], u0);
        fill_n(h_edge_buff_out[dir_x_2 + 1], dir_edge_sizes[dir], u0);

        // if edge proc => set buffers with given edge values
        if(!coords[dir]){
            fill_n(h_edge_buff_in[dir_x_2], dir_edge_sizes[dir], u[dir_x_2]);
        }
        if(coords[dir] == dimens[dir] - 1){
            fill_n(h_edge_buff_in[dir_x_2 + 1], dir_edge_sizes[dir], u[dir_x_2 + 1]);
        }
    }

    /* START OF ITTERATIVE COMPUTING: */
    double max_diff = 0.0; // maximum error
    thrust::device_ptr<double> i_ptr = thrust::device_pointer_cast(d_maxvalues);

    do{
        // Step 1(Send and recv all data):
        edges_exchange(h_edge_buff_in, h_edge_buff_out, dir_edge_sizes, 
            coords, dimens, neighb_ranks, grid_comm);

        // Step 2(copy edges to  device edges buffers)
        for(int orr = left; orr <= up; ++orr){
            cudaMemcpy(
                d_edge_buff_in[orr], 
                h_edge_buff_in[orr], 
                sizeof(double) * dir_edge_sizes[orr >> 1], 
                cudaMemcpyHostToDevice
            );
        }

        // Step 3(Copy and Fill inner edges)
        import_throught_x<<<LITTLE_BLOCKS, LITTLE_THREADS>>>(d_buffer0, d_edge_buff_in[left], 
                    d_edge_buff_in[right], blocks[dir_x], blocks[dir_y], blocks[dir_z]);
        import_throught_y<<<LITTLE_BLOCKS, LITTLE_THREADS>>>(d_buffer0, d_edge_buff_in[front], 
                    d_edge_buff_in[back], blocks[dir_x], blocks[dir_y], blocks[dir_z]);
        import_throught_z<<<LITTLE_BLOCKS, LITTLE_THREADS>>>(d_buffer0, d_edge_buff_in[down], 
                    d_edge_buff_in[up], blocks[dir_x], blocks[dir_y], blocks[dir_z]);
        
        cudaThreadSynchronize();


        // Step 4(compute new values):
        compute_new_grid<<<BLOCKS, THREADS>>>(d_buffer1, d_buffer0, d_maxvalues,
            blocks[dir_x], blocks[dir_y], blocks[dir_z], h2x, h2y, h2z);

        cudaThreadSynchronize();

        // Step 5(Fill out edges)
        export_throught_x<<<LITTLE_BLOCKS, LITTLE_THREADS>>>(d_buffer1, d_edge_buff_out[left], 
                    d_edge_buff_out[right], blocks[dir_x], blocks[dir_y], blocks[dir_z]);
        export_throught_y<<<LITTLE_BLOCKS, LITTLE_THREADS>>>(d_buffer1, d_edge_buff_out[front], 
                    d_edge_buff_out[back], blocks[dir_x], blocks[dir_y], blocks[dir_z]);
        export_throught_z<<<LITTLE_BLOCKS, LITTLE_THREADS>>>(d_buffer1, d_edge_buff_out[down], 
                    d_edge_buff_out[up], blocks[dir_x], blocks[dir_y], blocks[dir_z]);

        cudaThreadSynchronize();

        // Step 6(copy edges to  device edges buffers)
        for(int orr = left; orr <= up; ++orr){
            cudaMemcpy(
                h_edge_buff_out[orr], 
                d_edge_buff_out[orr], 
                sizeof(double) * dir_edge_sizes[orr >> 1], 
                cudaMemcpyDeviceToHost
            );
        }

        max_diff = *thrust::max_element(i_ptr, i_ptr + BLOCKS*THREADS);

        // Step 7(Control of stopping):
        MPI_Allgather(&max_diff, 1, MPI_DOUBLE, norm_data, 1, MPI_DOUBLE, grid_comm);

        max_diff = 0.0;
        for(int i = 0; i < workers_count; ++i){
            max_diff = max_diff < norm_data[i] ? norm_data[i] : max_diff; // maximum
        }


        // Step 6(swap buffers):
        double* tmp = d_buffer1;
        d_buffer1 = d_buffer0;
        d_buffer0 = tmp;
    }while(max_diff >= eps);

    // copy back to host:
    cudaMemcpy(h_buffer0, d_buffer0, sizeof(double) * buff_size, cudaMemcpyDeviceToHost);


    /* MPI I-O */
    //////////////////////////////////////////////////////////////////////////////////////////
    // convert data to txt
    char* write_data = new char[buff_size * float_size]; // data for human readable i-o
    //memset(write_data, (int)' ', buff_size * float_size * sizeof(char));
    for(int k = 1; k <= blocks[dir_z]; ++k){
        for(int j = 1; j <= blocks[dir_y]; ++j){
            int i, len;
            for(i = 1; i < blocks[dir_x]; ++i){
                len = sprintf(&write_data[idx(i, j, k)*float_size], "%.*e ", precission, h_buffer0[idx(i, j, k)]);
                // if we writed without minnus => we should change '\0' to separator
                if(len < float_size){
                    write_data[idx(i, j, k)*float_size + len] = ' ';
                }
            }
            // [OPTIONAL]: add '\n' to end instead ' '
            len = sprintf(&write_data[idx(i, j, k)*float_size], "%.*e\n", precission, h_buffer0[idx(i, j, k)]);
            if(len < float_size){
                write_data[idx(i, j, k)*float_size + len] = '\n';
            }
        }
    }

    MPI_Datatype float_represent;
    MPI_Type_contiguous(float_size, MPI_CHAR, &float_represent); 
    MPI_Type_commit(&float_represent); 
    //////////////////////////////////////////////////////////////////////////////////////////

    MPI_Datatype local_array, gloabal_array;
    int sizes[ndims], starts[ndims], gsizes[ndims], gstarts[ndims];

    sizes[dir_x] = size_x; sizes[dir_y] = size_y; sizes[dir_z] = size_z;
    starts[dir_x] = starts[dir_y] = starts[dir_z] = 1;

    gsizes[dir_x] = blocks[dir_x] * dimens[dir_x];
    gsizes[dir_y] = blocks[dir_y] * dimens[dir_y];
    gsizes[dir_z] = blocks[dir_z] * dimens[dir_z];

    gstarts[dir_x] = blocks[dir_x] * coords[dir_x];
    gstarts[dir_y] = blocks[dir_y] * coords[dir_y];
    gstarts[dir_z] = blocks[dir_z] * coords[dir_z];

    // create types for writing
    MPI_Type_create_subarray(3, sizes, blocks, starts, MPI_ORDER_FORTRAN, float_represent, &local_array); // memtype
    MPI_Type_create_subarray(3, gsizes, blocks, gstarts, MPI_ORDER_FORTRAN, float_represent, &gloabal_array); // filetype
    MPI_Type_commit(&local_array);
    MPI_Type_commit(&gloabal_array);

    // create and open file
    MPI_File fh;
    MPI_File_delete(path.c_str(), MPI_INFO_NULL);
    MPI_File_open(grid_comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    

    MPI_File_set_view(fh, 0, MPI_CHAR, gloabal_array, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, write_data, 1, local_array, MPI_STATUS_IGNORE);
    
    MPI_File_close(&fh);


    ///////////////////////////////////////////////////////////////////////////////////////

    /* END OF PROGRAM */
    // close Mpi connection
    MPI_Finalize();

    // free all memory
    delete[] norm_data;
    delete[] h_buffer0;
    delete[] write_data;
 
    cudaFree(d_buffer0);
    cudaFree(d_buffer1);
    cudaFree(d_maxvalues);

    for(int dir = 0; dir < ndims; ++dir){
        int dir_x_2 = dir << 1;
        delete[] h_edge_buff_out[dir_x_2];
        cudaFree(d_edge_buff_out[dir_x_2]);
        delete[] h_edge_buff_in[dir_x_2];
        cudaFree(d_edge_buff_in[dir_x_2]);
        delete[] h_edge_buff_out[dir_x_2 + 1];
        cudaFree(d_edge_buff_out[dir_x_2 + 1]);
        delete[] h_edge_buff_in[dir_x_2 + 1];
        cudaFree(d_edge_buff_in[dir_x_2 + 1]);
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