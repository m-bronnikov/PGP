// Made by Max Bronnikov
#ifndef __SCENE_CUH__
#define __SCENE_CUH__

#include <cstdio>
#include <stdexcept>
#include "figures.cuh"
#include "structures.cuh"
#include "ray_cleaner.cuh"
#include "file_writer.cuh"
#include "camera.cuh"
#include "material_table.cuh"

#define TRACE_BLOCKS 256
#define TRACE_THREADS 256
 
using namespace std;


class Scene{
private:
    // parametrs of window
    struct Window{
        uint32_t width;
        uint32_t height;
        uint32_t sqrt_scale;
        uint32_t* d_pixels; // data for write after ssaa
    };

    // parametrs of texture
    struct Texture{
        uint32_t width;
        uint32_t height;
        cudaArray* a_data;
        float_3 A;
        float_3 B;
        float_3 C;
        float_3 D;
        float_3 color; 
    };

    // Q. May be better to introduce structure for recursion memory incapsulation?

public:
    // init scene with viewer param and savepath
    Scene(const Camera& cam, const FileWriter& fout) 
    : viewer(cam), writter(fout) {
        // set sizes to 0:
        window.width = text.height = 0;
        text.width = text.height = 0;
    }

    void add_figure(const Figure3d& fig){
        fig.render_figure(render_triangles);
    }

    void add_light(const light_point& point){
        render_lights.push_back(point);
    }

    void set_window(uint32_t w, uint32_t h, uint32_t px_rays){
        window.width = w;
        window.height = h;
        window.sqrt_scale = px_rays;
    }

    void set_texture(const string& t_path, float_3 pA, float_3 pB, float_3 pC, float_3 pD, const material& t_mat){
        // TODO
    }

    void gpu_render_scene(const uint32_t recursion_depth = 1){
        if(window.width == 0 || window.height == 0 /*&& text check*/){
            throw std::runtime_error("Please set texture and window parametrs first!");
        }

        // size of upscaled image for ssaa aplication to bigger image
        uint32_t big_w = window.width * window.sqrt_scale;
        uint32_t big_h = window.height * window.sqrt_scale;

        float_3* d_img; // this rays colors - upscaled image for ssaa (summary colors for each source ray)
        triangle* d_triangles; // scene triangles
        light_point* d_lights; // scene lights
        recursion* d_rec; // data for recursion realisation
        material* d_mats;

        // get vector of materials
        MaterialTable().save_to_vector(render_maters);

        // Scene parameters:
        cout << "Scene Parametrs" << endl;
        cout << "Image size: " << window.width << "x" << window.height << endl;
        cout << "Rendering size: " << big_w << "x" << big_h << endl;
        cout << "Rays: " << big_w*big_h << endl;
        cout << "Triangles: " << render_triangles.size() << endl;
        cout << "Lights: " << render_lights.size() << endl;


        uint32_t allocated = 0;

        // Memory allocation:
        throw_on_cuda_error(cudaMalloc((void**)&d_triangles, render_triangles.size()*sizeof(triangle)));
        allocated += render_triangles.size()*sizeof(triangle);
        throw_on_cuda_error(cudaMemcpy(
            d_triangles, render_triangles.data(), 
            sizeof(triangle)*render_triangles.size(), 
            cudaMemcpyHostToDevice
        ));

        throw_on_cuda_error(cudaMalloc((void**)&d_lights, render_lights.size()*sizeof(light_point)));
        allocated += render_lights.size()*sizeof(light_point);
        throw_on_cuda_error(cudaMemcpy(
            d_lights, render_lights.data(), 
            sizeof(light_point)*render_lights.size(), 
            cudaMemcpyHostToDevice
        )); 

        // TODO: Set texture here

        throw_on_cuda_error(cudaMalloc((void**)&d_mats, render_maters.size()*sizeof(material)));
        allocated += render_maters.size()*sizeof(material);
        throw_on_cuda_error(cudaMemcpy(
            d_mats, render_maters.data(), 
            sizeof(material)*render_maters.size(), 
            cudaMemcpyHostToDevice
        ));

        // In order to econom memory, we could move some allocs and deallocs here to the loop
        throw_on_cuda_error(cudaMalloc((void**)&window.d_pixels, window.height*window.width*sizeof(uint32_t)));
        allocated += window.width*window.height*sizeof(uint32_t);

        throw_on_cuda_error(cudaMalloc((void**)&d_img, big_w*big_h*sizeof(float_3)));
        allocated += big_w*big_h*sizeof(float_3);

        // set capacity of allocated data to minimal size for first itteration:
        uint32_t rec_capacity = 2*big_w*big_h;
        throw_on_cuda_error(cudaMalloc((void**)&d_rec, rec_capacity*sizeof(recursion)));
        allocated += rec_capacity * sizeof(recursion);

        cout << "------------------------------------------------------" << endl;
        cout << "Allocated before execution: " << allocated / 1024 / 1024 << "Mb" << endl;
        cout << "------------------------------------------------------" << endl;

        // main loop
        cout << endl << "Render Starting" << endl;
        for(uint32_t frame_num = 0; viewer.update_position(); ++frame_num){
            cout << "=======================" << endl;
            cout << "Start render frame №" << frame_num << endl;
            mat_3x3 cam_matrix = viewer.get_frame_basis(); 
            float cam_z = viewer.get_z_coord();
            float_3 cam_pos = viewer.get_camera_position();

            uint32_t rec_size = big_h*big_w; 

            // init start values
            init_vewer_back_rays<<<TRACE_BLOCKS, TRACE_THREADS>>>(
                d_rec, d_img, cam_matrix, cam_pos, cam_z, big_w, big_h
            );
            cudaThreadSynchronize();
            throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

            cout << "Rays initialization done!" << endl;
            
            // loop of recursion here:
            for(uint32_t _ = 0; _ < recursion_depth && rec_size; ++_){
                cout << "running recursion depth №" << _ << endl;
                // kernel launch:
                // TODO: add texture to call
                gpu_ray_trace<<<TRACE_BLOCKS, TRACE_THREADS>>>(
                    d_rec, rec_size, d_img, d_mats,
                    d_triangles, render_triangles.size(),
                    d_lights, render_lights.size(),
                    big_w, big_h
                );
                cudaThreadSynchronize();
                throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

                cout << "cleaning rays(" << rec_size << ")" << endl;
                rec_size = cuda_clean_rays(d_rec, rec_size); // count of alive light rays

                // reallocation of memory for recursion:
                if(rec_capacity < (rec_size << 1)){
                    cout << "Here" << endl;
                    uint32_t was = allocated;
                    allocated -= rec_capacity * sizeof(recursion);

                    recursion* d_temp;
                    rec_capacity = rec_size << 1;

                    throw_on_cuda_error(cudaMalloc((void**)&d_temp, rec_capacity*sizeof(recursion)));
                    allocated += rec_capacity * sizeof(recursion);
                    throw_on_cuda_error(cudaMemcpy(d_temp, d_rec, sizeof(recursion)*rec_size, cudaMemcpyDeviceToDevice));
                    throw_on_cuda_error(cudaFree(d_rec));
                    d_rec = d_temp;

                    cout << "------------------------------------------------------" << endl;
                    cout << "Memory reallocated in runtime ";
                    cout << "from " << was / 1024 /1024 << "Mb to " << allocated / 1024 / 1024 << "Mb" << endl;
                    cout << "------------------------------------------------------" << endl;
                }
            }
            // TODO: SSAA here

            cout << "End render frame №" << frame_num << endl << endl;
            
            cout << "Start write frame №" << frame_num << " to the file" << endl;
            writter.write_to_file(window.d_pixels, frame_num); // write result to file
            cout << "End write frame №" << frame_num << " to the file" << endl;
            cout << "=======================" << endl;
        }

        throw_on_cuda_error(cudaFree(window.d_pixels));
        throw_on_cuda_error(cudaFree(d_img));
        throw_on_cuda_error(cudaFree(d_triangles));
        throw_on_cuda_error(cudaFree(d_lights));
        throw_on_cuda_error(cudaFree(d_mats));
        throw_on_cuda_error(cudaFree(d_rec));
    }

    ~Scene() = default;

private:
    FileWriter writter;
    Camera viewer;
    Window window;
    Texture text;

    vector<triangle> render_triangles;
    vector<light_point> render_lights;
    vector<material> render_maters;

    string path_to_dir;
};


#endif // __SCENE_CUH__