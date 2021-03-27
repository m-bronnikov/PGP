// Made by Max Bronnikov
#ifndef __SCENE_CUH__
#define __SCENE_CUH__

#include <cstdio>
#include <stdexcept>
#include <stdlib.h>
#include <fstream>
#include <chrono>
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
using namespace std::chrono;

/*
    Note: Class of scene for rendering

    This class described scene and include all elements of scene and methods for woking with it.
*/
class Scene{
public: 
    // init scene with viewer param, savepath and backend
    Scene(Camera& cam, FileWriter& fout, ostream& log_stream, bool is_gpu) 
    : viewer(cam), writter(fout), logger(log_stream), gpu_backend(is_gpu) {
        // set sizes and descriptors to 0 in order to check window and textures are setted:
        window.width  = 0;
        window.height = 0;
        floor.memory_wrapper = 0;
    }
private:
    // Window - parameters of image 
    struct Window{
        uint32_t width;
        uint32_t height;
        uint32_t sqrt_scale;
        uint32_t* device_picture; // finall picture
    };

    /*
        Note: Texture object creation

        Here we are read data of texture from file, create corresponding array 
        on GPU for this data, copy data to array and bind this array to texture 
        memory wrapper for better hardware perfomace of memory access.

        Method updates amount of allocated memory and returns pointer to data.
    */
    cudaArray* gpu_create_floor(uint32_t& allocated){
        // define array of data
        cudaArray* device_array;
        {
            ifstream floor_fin(path_to_floor, ios::in | ios::binary);
            if(not floor_fin){
                throw runtime_error("Floor opening broken!");
            }

            // image parameters
            uint32_t width, height;
            floor_fin.read(reinterpret_cast<char*>(&width), sizeof(uint32_t));
            floor_fin.read(reinterpret_cast<char*>(&height), sizeof(uint32_t));

            uint32_t size = width*height*sizeof(uint32_t);

            // allocation of data
            cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
            throw_on_cuda_error(cudaMallocArray(&device_array, &channel_desc, width, height));
            allocated += size;

            uint32_t* temp_data = new uint32_t[size];
            
            // read of data
            floor_fin.read(reinterpret_cast<char*>(temp_data), size);

            // copy data to cuda array
            throw_on_cuda_error(cudaMemcpyToArray(device_array, 0, 0, temp_data, size, cudaMemcpyHostToDevice));

            // free temp data
            delete[] temp_data;
        }

        // description of texture data
        struct cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = device_array;

        // description of texture properties
        struct cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;

        // create texture data wrapper
        floor.memory_wrapper = 0;
        throw_on_cuda_error(cudaCreateTextureObject(&floor.memory_wrapper, &res_desc, &tex_desc, 0));

        return device_array;
    }

    /*
        Note: Destroy floor memory object.
    */
    void gpu_destroy_floor(){
        cudaDestroyTextureObject(floor.memory_wrapper);
        floor.memory_wrapper = 1;
    }

public:
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

    void set_floor(const string& floor_path, float_3 pA, float_3 pB, float_3 pC, float_3 pD, const material& floor_material){
        // define id of material
        uint32_t material_id = MaterialTable().get_material_id(floor_material);
        // set triangles
        floor.id_of_triangle_1 = render_triangles.size();
        render_triangles.push_back({pA, pB, pD, material_id});

        floor.id_of_triangle_2 = render_triangles.size();
        render_triangles.push_back({pC, pB, pD, material_id});
        
        // textures are setted
        floor.memory_wrapper = 1;

        // save path to texture
        path_to_floor = floor_path;
    }

    /*
        Note: Render video of scene

        This is main method of this project, which generates video of scene by frame.
        Here we allocate memory of GPU and run computations of ray tracing. 
    */
    void gpu_render_scene(const uint32_t recursion_depth = 1){
        // check texture and window parameter setted
        if(window.width == 0 || window.height == 0 || floor.memory_wrapper == 0){
            throw std::runtime_error("Please set floor and window parametrs first!");
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

        // Scene parameters:
        logger << "SCENE PARAMETERS" << endl << endl;
        logger << "Image size: " << window.width << "x" << window.height << endl;
        logger << "Rendering size: " << scaled_w << "x" << scaled_h << endl;
        logger << "Rays: " << scaled_w*scaled_h << endl;
        logger << "Triangles: " << render_triangles.size() << endl;
        logger << "Lights: " << render_lights.size() << endl;


        // Memory allocation:
        uint32_t allocated = 0;

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
        
        throw_on_cuda_error(cudaMalloc((void**)&device_materials, render_maters.size()*sizeof(material)));
        allocated += render_maters.size()*sizeof(material);
        throw_on_cuda_error(cudaMemcpy(
            device_materials, render_maters.data(), 
            sizeof(material)*render_maters.size(), 
            cudaMemcpyHostToDevice
        ));

        cudaArray* floor_array = gpu_create_floor(allocated);

        // In order to econom memory, we could move some allocs and deallocs here to the loop
        throw_on_cuda_error(cudaMalloc((void**)&window.device_picture, window.height*window.width*sizeof(uint32_t)));
        allocated += window.width*window.height*sizeof(uint32_t);

        throw_on_cuda_error(cudaMalloc((void**)&device_img, scaled_w*scaled_h*sizeof(float_3)));
        allocated += scaled_w*scaled_h*sizeof(float_3);

        // set capacity of allocated data to minimal size for first itteration:
        uint32_t active_rays_capacity = 2*scaled_w*scaled_h;
        throw_on_cuda_error(cudaMalloc((void**)&device_rays_data, active_rays_capacity*sizeof(recursion)));
        allocated += active_rays_capacity * sizeof(recursion);

        logger << "------------------------------------------------------" << endl;
        logger << "Allocated before execution: " << allocated / 1024 / 1024 << "Mb" << endl;
        logger << "------------------------------------------------------" << endl;

        // main loop
        logger << endl << "RENDERING" << endl << endl;
        for(uint32_t number_of_frame = 0; viewer.update_position(); ++number_of_frame){
            logger << "=======================" << endl;
            logger << "Rendering frame №" << number_of_frame << endl;
            auto time_start = steady_clock::now();

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
            
            // Loop of recursion here:
            for(uint32_t _ = 0; _ < recursion_depth && active_rays_size; ++_){
                logger << "\tRecursive depth №" << _ << " use " << active_rays_size << " rays" << endl;

                // Kernel launch:
                {
                    gpu_ray_trace<<<TRACE_BLOCKS, TRACE_THREADS>>>(
                        device_rays_data, active_rays_size, device_img, device_materials,
                        floor, device_triangles, render_triangles.size(),
                        device_lights, render_lights.size(),
                        scaled_w, scaled_h
                    );
                    cudaThreadSynchronize();
                    throw_on_cuda_error(cudaGetLastError()); // catch errors from kernel

                    // Kernell potentialy can produce x2 rays count:
                    active_rays_size <<= 1;
                }

                // Clean all dead rays from array of active rays:
                active_rays_size = cuda_clean_rays(device_rays_data, active_rays_size);  

                // Reallocation of memory for recursion if needed:
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

                    logger << "------------------------------------------------------" << endl;
                    logger << "Memory reallocated in runtime ";
                    logger << "from " << was / 1024 /1024 << "Mb to " << allocated / 1024 / 1024 << "Mb" << endl;
                    logger << "------------------------------------------------------" << endl;
                }
            }

            // Call SSAA for anti-aliasing effect.
            ssaa<<<dim3(TRACE_BLOCKS_2D, TRACE_BLOCKS_2D), dim3(TRACE_THREADS_2D, TRACE_THREADS_2D)>>>(
                window.device_picture, device_img, window.width, window.height, window.sqrt_scale
            );
            cudaThreadSynchronize();

            writter.write_to_file(window.device_picture, number_of_frame); // write result to file

            auto time_end =  steady_clock::now();
            logger << "Time of frame rendering: ";
            logger << (duration_cast<microseconds>(time_end - time_start).count() / 1000.0) << endl;
            logger << "=======================" << endl;
        }

        gpu_destroy_floor();
        throw_on_cuda_error(cudaFreeArray(floor_array));
        throw_on_cuda_error(cudaFree(window.device_picture));
        throw_on_cuda_error(cudaFree(device_img));
        throw_on_cuda_error(cudaFree(device_triangles));
        throw_on_cuda_error(cudaFree(device_lights));
        throw_on_cuda_error(cudaFree(device_materials));
        throw_on_cuda_error(cudaFree(device_rays_data));
    }

    ~Scene() = default;

private:
    bool gpu_backend;
    ostream& logger;
    
    FileWriter& writter;
    Camera& viewer;
    Window window;

    vector<triangle> render_triangles;
    vector<light_point> render_lights;
    vector<material> render_maters;
    gpu_texture floor;

    string path_to_dir;
    string path_to_floor;
};


#endif // __SCENE_CUH__