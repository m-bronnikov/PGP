// Made by Max Bronnikov
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <math.h>
#include "mpi.h"
#include "mpi_io.cuh"
#include "figures.cuh"
#include "structures.cuh"
#include "ray_tracing.cuh"
#include "scene.cuh"

using namespace std;

string help_message(){
    stringstream out;
    out << "Usage: ./run [arg]" << endl;
    out << endl << "`arg` may be:" << endl;
    out << "1. `--gpu` for set Nvidia GPU as backend" << endl;
    out << "2. `--cpu` for set CPU as backend" << endl;
    out << "3. `--default` for print deafult configuration and exit" << endl;
    out << "4. `--help` for print this message and exit" << endl;
    out << endl << "Made by Max Bronnikov" << endl;

    return out.str();
}

string best_configuration(){
    stringstream out;
    out << "The best configuration:" << endl;
    // frames
    out << "104" << endl;
    // save path
    out << "data/%d.data" << endl;
    // w, h, angle
    out << "720 480 120" << endl;
    // camera
    out << "4.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0" << endl;
    out << "1.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0" << endl;
    // figures
    out << "2.0 0.0 0.0 1.0 0.0 0.0 1.25 0.9 0.5 10" << endl;
    out << "0.0 2.0 0.0 0.0 1.0 0.0 1.0 0.8 0.5 5" << endl;
    out << "0.0 0.0 0.0 0.0 0.7 0.7 0.85 0.7 0.3 5" << endl;
    // texture
    out << "-5.0 -5.0 -1.0 -5.0 5.0 -1.0 5.0 5.0 -1.0 5.0 -5.0 -1.0" << endl;
    out << "textures/floor.data 0.0 1.0 0.0 0.5" << endl;
    // lights
    out << "2" << endl;
    out << "-10.0 0.0 10.0 1.0 1.0 1.0" << endl;
    out << "1.0 0.0 10.0 1.0 0.0 1.0" << endl;
    // recursion and avg_pool winow size
    out << "6 2" << endl;

    return out.str();
}


int main(int argc, char** argv){
    // MPI Init with reader
    int proc_rank, num_of_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    MpiReader is(cin, proc_rank);

    // CUDA GPU Init by procs
    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(proc_rank % device_count);

    // Define backend and work with user parameter
    bool is_gpu = true;
    if(argc == 2){
        string arg = argv[1];

        if(arg == "--gpu"){
            is_gpu = true;
        }else if(arg == "--cpu"){
            is_gpu = false;
        }else if(arg == "--deafult"){
            cout << best_configuration() << endl;
            return 0;
        }else if(arg == "--help" || arg == "-h"){
            cout << help_message() << endl;
            return 0;
        }else{
            throw invalid_argument("Unknown command line parameter!");
        }
    }else if(argc > 2){
        cerr << help_message() << endl;
        throw invalid_argument("Wrong command line parameters count!");
    }

    // Read properties of rendering scene
    {
        uint32_t frames;

        string path_modifier;

        uint32_t width, height;
        float view_angle;

        float rc_0, zc_0, fc_0, Ac_r, Ac_z, wc_r, wc_z, wc_f, pc_r, pc_z;
        float rn_0, zn_0, fn_0, An_r, An_z, wn_r, wn_z, wn_f, pn_r, pn_z;

        float_3 center, color;
        float radius;
        float reflection, refraction;
        uint32_t line_lights;

        float_3 floor_A, floor_B, floor_C, floor_D;
        string texture_path;
        float_3 texture_color;
        float texture_refl;

        uint32_t light_num;

        uint32_t recursion_depth, sqrt_per_pixel;

        // read:
        // 1.
        is >> frames;

        // 2.
        is >> path_modifier;

        // 3.
        is >> width >> height;
        is >> view_angle;

        // 4.
        is >> rc_0 >> zc_0 >> fc_0 >> Ac_r >> Ac_z >> wc_r >> wc_z >> wc_f >> pc_r >> pc_z;
        is >> rn_0 >> zn_0 >> fn_0 >> An_r >> An_z >> wn_r >> wn_z >> wn_f >> pn_r >> pn_z;
        // create main objects
        FileWriter writter(width, height, path_modifier, is_gpu);
        Camera camera(
            frames, proc_rank, num_of_procs, view_angle, 
            rc_0, zc_0, fc_0, Ac_r, Ac_z, wc_r, wc_z, wc_f, pc_r, pc_z,
            rn_0, zn_0, fn_0, An_r, An_z, wn_r, wn_z, wn_f, pn_r, pn_z
        );
        
        Scene scene(camera, writter, MpiLogger(cout, proc_rank, num_of_procs), is_gpu);

        // 5.
        // Tetraeder:
        is >> center.x >> center.y >> center.z;
        is >> color.x >> color.y >> color.z;
        is >> radius;
        is >> reflection >> refraction;
        is >> line_lights; 
        material fig_glass = {color, 0.5, reflection, refraction}; // set diffusion as 0.5 by default
        scene.add_figure(Tetraeder(radius, center, fig_glass, line_lights));

        // Octaeder:
        is >> center.x >> center.y >> center.z;
        is >> color.x >> color.y >> color.z;
        is >> radius;
        is >> reflection >> refraction;
        is >> line_lights; 
        fig_glass = {color, 0.5, reflection, refraction}; // set diffusion as 0.5 by default
        scene.add_figure(Octaeder(radius, center, fig_glass, line_lights));

        // Dodecaedr:
        is >> center.x >> center.y >> center.z;
        is >> color.x >> color.y >> color.z;
        is >> radius;
        is >> reflection >> refraction;
        is >> line_lights; 
        fig_glass = {color, 0.5, reflection, refraction}; // set diffusion as 0.5 by default
        scene.add_figure(Dodecaedr(radius, center, fig_glass, line_lights));

        // 6.
        is >> floor_A.x >> floor_A.y >> floor_A.z;
        is >> floor_B.x >> floor_B.y >> floor_B.z;
        is >> floor_C.x >> floor_C.y >> floor_C.z;
        is >> floor_D.x >> floor_D.y >> floor_D.z;
        is >> texture_path;
        is >> texture_color.x >> texture_color.y >> texture_color.z;
        is >> texture_refl;
        material texture_mat = {texture_color, 0.5, texture_refl, 0.0}; // setd diffusion as 1.0

        // 7.
        is >> light_num;
        for(uint8_t i = 0; i < light_num; ++i){
            float_3 light_coord, light_color;
            is >> light_coord.x >> light_coord.y >> light_coord.z;
            is >> light_color.x >> light_color.y >> light_color.z;

            scene.add_light({light_coord, light_color, 1.0}); // set power as 1 by default
        }

        // 8.
        is >> recursion_depth >> sqrt_per_pixel;

        // set window before render:
        scene.set_window(width, height, sqrt_per_pixel);
        scene.set_floor(texture_path, floor_A, floor_B, floor_C, floor_D, texture_mat);

        // launch render ;)
        scene.render_scene(recursion_depth);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}