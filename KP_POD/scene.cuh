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
#define TRACE_BLOCKS_2D 32
#define TRACE_THREADS 256
#define TRACE_THREADS_2D 32
 
using namespace std;

class Scene{
private:
    // Window - parameters of image 
    struct Window{
        uint32_t width;
        uint32_t height;
        uint32_t sqrt_scale;
        uint32_t* device_picture; // finall picture
    };

    // Texture contain params of texture
    struct Texture{
        uint32_t width;
        uint32_t height;
        cudaArray* array_data;
        float_3 A;
        float_3 B;
        float_3 C;
        float_3 D;
        float_3 color; 
    };

    // Q. May be better to introduce structure for recursion memory incapsulation?

public:
    // init scene with viewer param and savepath
    Scene(Camera& cam, FileWriter& fout) 
    : viewer(cam), writter(fout) {
        // set sizes to 0:
        window.width = texture.height = 0;
        texture.width = texture.height = 0;
    }

    void add_figure(const Figure3d& fig){
        fig.render_figure(render_triangles, render_lights);
    }

    void add_light(const light_point& point){
        render_lights.push_back(point);
    }

    void set_window(uint32_t w, uint32_t h, uint32_t sqrt_pixel_rays){
        window.width = w;
        window.height = h;
        window.sqrt_scale = sqrt_pixel_rays;
    }

    void set_texture(const string& t_path, float_3 pA, float_3 pB, float_3 pC, float_3 pD, const material& texture_material){
        // TODO
    }

    void gpu_render_scene(const uint32_t recursion_depth = 1){
        if(window.width == 0 || window.height == 0 /*&& texture check*/){
            throw std::runtime_error("Please set texture and window parametrs first!");
        }

        // size of upscaled image for ssaa aplication to bigger image
        uint32_t scaled_w = window.width * window.sqrt_scale;
        uint32_t scaled_h = window.height * window.sqrt_scale;

        float_3* device_img; // this rays colors - upscaled image for ssaa (summary colors for each source ray)
        triangle* device_triangles; // scene triangles
        light_point* device_lights; // scene lights
        recursion* device_rays_data; // data for recursion realisation
        material* device_materials;

        // get vector of materials
        MaterialTable().save_to_vector(render_maters);

        for(int i = 0; i < render_maters.size(); ++i){
            cout << "Material #" << i << ":" << endl;
            cout << "diffusion" << render_maters[i].diffussion << endl;
            cout << "reflection" << render_maters[i].reflection << endl;
            cout << "refraction" << render_maters[i].refraction << endl;

            cout << "color: " << render_maters[i].color.x << " " << render_maters[i].color.y << " " << render_maters[i].color.z << endl; 
        }

        // Scene parameters:
        cout << "Scene Parametrs" << endl;
        cout << "Image size: " << window.width << "x" << window.height << endl;
        cout << "Rendering size: " << scaled_w << "x" << scaled_h << endl;
        cout << "Rays: " << scaled_w*scaled_h << endl;
        cout << "Triangles: " << render_triangles.size() << endl;
        cout << "Lights: " << render_lights.size() << endl;


        uint32_t allocated = 0;

        // Memory allocation:
        throw_on_cuda_error(cudaMalloc((void**)&device_triangles, render_triangles.size()*sizeof(triangle)));
        allocated += render_triangles.size()*sizeof(triangle);
        throw_on_cuda_error(cudaMemcpy(
            device_triangles, render_triangles.data(), 
            sizeof(triangle)*render_triangles.size(), 
            cudaMemcpyHostToDevice
        ));

        throw_on_cuda_error(cudaMalloc((void**)&device_lights, render_lights.size()*sizeof(light_point)));
        allocated += render_lights.size()*sizeof(light_point);
        throw_on_cuda_error(cudaMemcpy(
            device_lights, render_lights.data(), 
            sizeof(light_point)*render_lights.size(), 
            cudaMemcpyHostToDevice
        )); 

        // TODO: Set texture here
        throw_on_cuda_error(cudaMalloc((void**)&device_materials, render_maters.size()*sizeof(material)));
        allocated += render_maters.size()*sizeof(material);
        throw_on_cuda_error(cudaMemcpy(
            device_materials, render_maters.data(), 
            sizeof(material)*render_maters.size(), 
            cudaMemcpyHostToDevice
        ));

        // In order to econom memory, we could move some allocs and deallocs here to the loop
        throw_on_cuda_error(cudaMalloc((void**)&window.device_picture, window.height*window.width*sizeof(uint32_t)));
        allocated += window.width*window.height*sizeof(uint32_t);

        throw_on_cuda_error(cudaMalloc((void**)&device_img, scaled_w*scaled_h*sizeof(float_3)));
        allocated += scaled_w*scaled_h*sizeof(float_3);

        // set capacity of allocated data to minimal size for first itteration:
        uint32_t active_rays_capacity = 2*scaled_w*scaled_h;
        throw_on_cuda_error(cudaMalloc((void**)&device_rays_data, active_rays_capacity*sizeof(recursion)));
        allocated += active_rays_capacity * sizeof(recursion);

        cout << "------------------------------------------------------" << endl;
        cout << "Allocated before execution: " << allocated / 1024 / 1024 << "Mb" << endl;
        cout << "------------------------------------------------------" << endl;

        // main loop
        cout << endl << "Render Starting" << endl;
        for(uint32_t number_of_frame = 0; viewer.update_position(); ++number_of_frame){
            cout << "=======================" << endl;
            cout << "Start render frame №" << number_of_frame << endl;
            mat_3x3 transformation_matrix = viewer.get_frame_basis(); 
            float distance_to_viewer = viewer.get_distance_to_viewer();
            float_3 camera_position = viewer.get_camera_position();

            uint32_t active_rays_size = scaled_h*scaled_w; 

            // init start values
            init_vewer_back_rays<<<TRACE_BLOCKS, TRACE_THREADS>>>(
                device_rays_data, device_img, transformation_matrix, camera_position, distance_to_viewer, scaled_w, scaled_h
            );
            cudaThreadSynchronize();
            throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

            cout << "Rays initialization done!" << endl;
            
            // loop of recursion here:
            for(uint32_t _ = 0; _ < recursion_depth && active_rays_size; ++_){
                cout << "running recursion depth №" << _ << endl;
                // kernel launch:
                {
                    // TODO: add texture to call
                    gpu_ray_trace<<<TRACE_BLOCKS, TRACE_THREADS>>>(
                        device_rays_data, active_rays_size, device_img, device_materials,
                        device_triangles, render_triangles.size(),
                        device_lights, render_lights.size(),
                        scaled_w, scaled_h
                    );
                    cudaThreadSynchronize();
                    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

                    // Kernell potentialy can produce x2 rays count:
                    active_rays_size <<= 1;
                }

                // Clean all dead rays from array of active rays:
                cout << "cleaning rays(" << active_rays_size << ")" << endl;
                active_rays_size = cuda_clean_rays(device_rays_data, active_rays_size);  

                // reallocation of memory for recursion:
                if(active_rays_capacity < (active_rays_size << 1)){
                    uint32_t was = allocated;
                    allocated -= active_rays_capacity * sizeof(recursion);

                    recursion* temporary_device_rays;
                    active_rays_capacity = active_rays_size << 1;

                    throw_on_cuda_error(cudaMalloc((void**)&temporary_device_rays, active_rays_capacity*sizeof(recursion)));
                    allocated += active_rays_capacity * sizeof(recursion);
                    throw_on_cuda_error(cudaMemcpy(temporary_device_rays, device_rays_data, sizeof(recursion)*active_rays_size, cudaMemcpyDeviceToDevice));
                    throw_on_cuda_error(cudaFree(device_rays_data));
                    device_rays_data = temporary_device_rays;

                    cout << "------------------------------------------------------" << endl;
                    cout << "Memory reallocated in runtime ";
                    cout << "from " << was / 1024 /1024 << "Mb to " << allocated / 1024 / 1024 << "Mb" << endl;
                    cout << "------------------------------------------------------" << endl;
                }
            }

            cout << "End render frame №" << number_of_frame << endl << endl;

            cout << "Launch SSAA for image" << endl;
            ssaa<<<dim3(TRACE_BLOCKS_2D, TRACE_BLOCKS_2D), dim3(TRACE_THREADS_2D, TRACE_THREADS_2D)>>>(
                window.device_picture, device_img, window.width, window.height, window.sqrt_scale
            );
            cout << "SSAA Done successfully" << endl << endl;

            
            cout << "Start write frame №" << number_of_frame << " to the file" << endl;
            writter.write_to_file(window.device_picture, number_of_frame); // write result to file
            cout << "End write frame №" << number_of_frame << " to the file" << endl;
            cout << "=======================" << endl;
        }

        throw_on_cuda_error(cudaFree(window.device_picture));
        throw_on_cuda_error(cudaFree(device_img));
        throw_on_cuda_error(cudaFree(device_triangles));
        throw_on_cuda_error(cudaFree(device_lights));
        throw_on_cuda_error(cudaFree(device_materials));
        throw_on_cuda_error(cudaFree(device_rays_data));
    }

    ~Scene() = default;

private:
    FileWriter& writter;
    Camera& viewer;
    Window window;
    Texture texture;

    vector<triangle> render_triangles;
    vector<light_point> render_lights;
    vector<material> render_maters;

    string path_to_dir;
};


#endif // __SCENE_CUH__