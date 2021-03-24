// Made by Max Bronnikov
#ifndef __RAY_TRACING_CUH__
#define __RAY_TRACING_CUH__

#include "structures.cuh"

// infinity value:
#define MAX_FLOAT 1e+32
// precission of float ~1e-7:
#define EPSILON 1e-6 
// set as material parametr in future
#define SPEC_POW 64

/* 
    Note: Contract for intersection searching.
*/
struct search_request
{
    float distance;
    uint32_t id; // id of triangle
};

// TODO Define this for textures
//  texture<uint32_t, 2, cudaReadModeElementType> g_text;

/*
Note: This function define intersection with triangle and distanse "ts" to this.
      Exist two ways to perform this function:
        1. Baricentric test
        2. Unit test

    On GPU better to use second way, but here we use first variant because it 
    much more popular solution. 

    This function return negative value if test failed and "ts" - distance to triangle in other case.
*/
__device__
float triangle_intersected(const back_ray& ray, const triangle& current_triangle){
    float_3 e1 = current_triangle.b - current_triangle.a;
    float_3 e2 = current_triangle.c - current_triangle.a;
    float_3 p = cross(ray.dir, e2);
    float div_n = dot(p, e1);

    if (abs(div_n) < EPSILON){
        return -1.0f;
    }

    float_3 t = ray.from - current_triangle.a;

    float u = dot(p, t) / div_n;

    if(u < 0.0f || u > 0.0f){
        return -1.0f;
    }

    float_3 q = cross(t, e1);

    float v = dot(q, ray.dir) / div_n;

    // TODO: Investigate that if/else 3 times may slow our program, beter run if/else at ones?
    if(v >= 0.0f && v + u <= 1.0){
        return dot(q, e2) / div_n;
    }

    return -1.0f;
}

/*
Note: This function define nearest intersected by ray triangle from each intersected ones.

      If intersection doesn't exist, this function return false.
*/
__device__
bool search_triangle_intersection(
    search_request& result_contract, const back_ray& ray, const triangle* array_scene_triangles, uint32_t count_of_triangles
){
    // Init contract before search:
    result_contract.distance = MAX_FLOAT; // infinity
    result_contract.id = count_of_triangles; // imposible value to detect intersection fact later

    for(uint32_t i = 0; i < count_of_triangles; ++i){
        triangle current_triangle = array_scene_triangles[i];

		float ts = triangle_intersected(ray, current_triangle); 	

        // search intersection with minimal distance
		if(ts >= EPSILON && ts < result_contract.distance){
			result_contract.distance = ts;
			result_contract.id = i;
        }
    }

    // if triangles not intersected => id equal to count of triangles in array
    return result_contract.id < count_of_triangles;
}

/*
  Note: It is possible that we meat some triangles between intersection and light source.
        In classic ray tracing we set this shading to black(0, 0, 0) and go to compute next shade.
        But let me show following example:

                                  |         |
      (intersection) o------------|- - - - -| -  -  -  - 0  (light source)
                                  |         |
                            surface №1  surface №2
                           (refr1=0.8)  (refr2=0.6)

        In this case color, which come to intersection, will: 
        `radiation = color*(refr1*refr2)*light_power`

        This function computes product of refractions between intersection and light source. 
        It will return portion of each color component in incomed to intersection ray. 

        // TODO Make float if color components addition is extra.
*/
__device__
float_3 compute_radiocity_losses(
    const back_ray& ray, const material* scene_materials, const triangle* array_scene_triangles, 
    uint32_t count_of_triangles, uint32_t exclude_triangle_id, float distance_to_target
){
    float_3 power = {1.0, 1.0, 1.0};
    for(uint32_t i = 0; i < count_of_triangles; ++i){
        triangle barrier_triangle = array_scene_triangles[i];

		float ts = triangle_intersected(ray, barrier_triangle); 	

        // `i != exclude_triangle_id` helps to avoid intersections with start position. We can add 
        // this check in start of loop body, but it's rare case and better to check it latter. 
        // TODO optimize this (remove third check)
		if(ts >= EPSILON && ts < distance_to_target /* && i != exclude_triangle_id */){
            // TODO: Investigate gain from multipy to color of materaial. May be better to remove.
            material barrier_material = scene_materials[barrier_triangle.mat_id];
            // TODO Remove color mult if needed
            power *= barrier_material.color;
            power *= barrier_material.refraction;
        }
    }

    return power;
}


/*
  Note: This function computes ideal reflection of ray with surface.
        Formula: 
                    r = i - 2*n*dot(n, i)

                              n
                        \     |   
                         \   /|\ __/
                          \   |   /|
                        i  \  |  /   r
                           _\|| /     
                    _________\|/_____________
 
        i - incomming ray, r - reflected ray, n - normal.
*/
__device__
float_3 reflect(const float_3& incoming_dir, const float_3& normal){
    float_3 ref = normal;
    ref *= 2.0f * dot(normal, incoming_dir);
    ref -= incoming_dir;

    return -ref;
}


/*
    Note: This function corrects normal direction.

                     viewer   n (1)
                        \     |   
                         \   /|\ 
                          \   |  
                        i  \  |  
                           _\||      surface
                    _________\|_____________
                              |
                              |
                              |
                             \|/
                              | 
                              n (2)

    For correct shine computation normal of surface should be in position (1), 
    bur it's possible to meet both positions ((1) and (2)). This function corrects 
    normal to true direction.

    Condition:
    We should change direction of normal if dot(i, normal) > 0.0.
*/
__device__
float_3 correct_normal(const float_3& incoming_dir, const float_3& normal){
    return dot(incoming_dir, normal) > 0.0f ? -normal : normal;
}

/*
    Note: In ray tracing we do not use radiation from back side of surface (even if surface refract rays).
        This fact entails disadvantage of our implementaion and classic ray tracing. 
        Example:

                    viewer  normal
                        \     |   light №2
                         \   /|\ __/
                          \   |   /|
                      ray  \  |  /
                           _\|| /     Surface
                    _________\|/_____________
                             /
                            /
                          |/_
                          /
                      light №1

        In this case we can't compute shade for "light №1" and compute shade only by "light №2".

        Therefore, continue if dot(normal, ray_to_light) <= 0.0
*/
__device__
bool is_light_source_on_dark_side(const float_3& source_dir, const float_3& normal){
    return dot(source_dir, normal) < EPSILON;
}

/*
  Note: Phong Shading without phone. 
       Formula:
            lambert_coeff = diff_k * dot(normal, l_r)
            phong_coeff = lambert_coeff + spec_k * dot(reflect(l_r), v_r)^pow

            shade = ray_power * source_color * source_power * material_color * phong_coeff

        l_r - ray to light, v_r - ray to viewer, pow - some big number(equal to 32 here),
        diff_k, spec_k - diffusion and specular coeffs of material respectively.
  
  In this work I use coefficent of specular shade equal to reflection coeff.

  It's possible to use only Lambert shading model.

  TODO: Investigate gain from usge this model and correct of using specular == reflections coeffs.
*/
__device__
float_3 phong_shading(
    const float_3& normal, const float_3& ray_to_viewer, const float_3& ray_to_light, 
    const light_point& shine_source, const material material_properties, const float_3& radiocity_portion
){
    // diffusion
    float phong_coeff = material_properties.diffussion * dot(normal, ray_to_light);
    // specular
    phong_coeff += material_properties.reflection * max(0.0, pow(dot(reflect(ray_to_light, normal), ray_to_viewer), SPEC_POW));

    // multiplication of colors with portion factor
    float_3 color = shine_source.color * material_properties.color * radiocity_portion;

    // final shade
    return color * shine_source.power * phong_coeff;
}

/*
Main Idea:

At first step we throw rays to each pixel. After that we search intersections 
with some triangles (if not => black color). In place of intersection we 
shade our ray (and pixel respectivetely). 

Shading include:
    1. Direct lights from all light sources. We use Phong shading for this.
    2. Color from shaded reflected ray.
    3. Color from shaded refracted ray.

We can compute it recursive with following solution:
    0. Input data of recursion - ray. 
    1. Find intersection for ray. If intersection not exist - return.
    2. Compute step №1 of shading strategy and add obtained color to summary color of binded pixel.
    3. Search reflection and refraction rays(if exist), bind them to current pixel and 
       run this method recursive for these rays.

It possible because shading of ray is associative operation.
*/
__global__
void gpu_ray_trace(
    recursion* array_of_rays_info, uint32_t active_rays_count,
    float_3* image, material* scene_materials,
    triangle* array_scene_triangles, uint32_t count_of_triangles,
    light_point* array_of_light_points, uint32_t count_of_light_points,
    uint32_t w, uint32_t h
){
    uint32_t thread_step = blockDim.x * gridDim.x;
    uint32_t thread_start = threadIdx.x + blockIdx.x*blockDim.x;

    /*
        Note: We compute ray tracing for each recursive ray by parallel. 
              Each ray binded to some pixel.
    */
    for(uint32_t info_id = thread_start; info_id < active_rays_count; info_id += thread_step){
        recursion current_ray_info = array_of_rays_info[info_id];

        // Create search contract with `id` of nearest intersected ray and `distance` to intersection
        search_request intersected_triangle_data;
    
        // If intersection not occured => make reflected and refracted rays as "dead" and finish computation for this ray.
        if(not search_triangle_intersection(intersected_triangle_data, current_ray_info.ray, array_scene_triangles, count_of_triangles)){
            array_of_rays_info[info_id].power = 0.0f;
            array_of_rays_info[active_rays_count + info_id].power = 0.0f;
            continue;
        }

        // Extract info about intersected triangle from gloabal memory.
        float_3 triangle_normal;
        material material_properties;
        {
            triangle found_triangle = array_scene_triangles[intersected_triangle_data.id]; // intersected triangle

            material_properties = scene_materials[found_triangle.mat_id];
            triangle_normal = correct_normal(current_ray_info.ray.dir, normal_of_triangle(found_triangle));
        }

        // Jump to intersection
        float_3 intersection_position = current_ray_info.ray.from + current_ray_info.ray.dir*intersected_triangle_data.distance;

        /*
        Note: Search shading color by all lights. 
        
              Here we throw rays from intersection  to all light sources. If ray comes to target - 
              compute shading by this light with Phong shading model(diffuse + specular, without phone). 

              In this work I use coefficent of specular shade equal to reflection coeff.

              TODO: Investigate gain from usge this model and correct of using specular == reflections coeffs.
        */
        float_3 overall_lights_shine = {0.0, 0.0, 0.0}; 

        for(uint32_t point_id = 0; point_id < count_of_light_points; ++point_id){
            // Get light_source and create ray to this point^
            light_point source_of_light = array_of_light_points[point_id];
            float_3 vector_to_source = source_of_light.coord - intersection_position;
            back_ray ray_to_light_source = back_ray{intersection_position, norm(vector_to_source)};

            if(is_light_source_on_dark_side(ray_to_light_source.dir, triangle_normal)){
                continue;
            }

            /*
             Note: Define color of light ray, which came to our intersection from source.

            TODO: Remove source_of_light.power from this program.
            */
            {
                // Define portion of light incomed to intersection
                float_3 incomed_shine_loss = compute_radiocity_losses(
                    ray_to_light_source, scene_materials, array_scene_triangles, count_of_triangles, intersected_triangle_data.id, abs(vector_to_source)
                );

                /*
                Note: Add shading computed with Phong model to result shine
                
                 In Phong shading used ray to viewer instead ray from viewer.
                 It reason to use `-current_ray_info.ray.dir` as a param of this function. 
                */
                overall_lights_shine += phong_shading(
                    triangle_normal, -current_ray_info.ray.dir, ray_to_light_source.dir, source_of_light, material_properties, incomed_shine_loss
                );
            }
        }

        // Contribution of lights must be corrected with current ray power:
        overall_lights_shine *= current_ray_info.power;
        
        /*
            Note: Write result to image in global memory.
            
            Many rays may try to write shine to target image. Let's use `atomicAdd` for this! 
        */
        atomicAdd(&image[current_ray_info.y * w + current_ray_info.x], overall_lights_shine); 

        /*
        Note: Produce new rays. This part pushing new rays:
                1. Reflected from current ray from viewer with lower power of ray.
                2. Refracted, which continue to pass in current direction, but with lower power.

                After that this part emplace this rays to recursion array.

                Requirements: direction of ray should be always normilized.
        */
        {
            // reflection
            recursion reflected_ray_info = current_ray_info;
            reflected_ray_info.power *= material_properties.reflection;
            reflected_ray_info.ray.from = intersection_position;
            reflected_ray_info.ray.dir = norm(reflect(current_ray_info.ray.dir, triangle_normal));


            // refraction
            current_ray_info.ray.from = intersection_position;
            current_ray_info.power *= material_properties.refraction;

            // push back to memory
            array_of_rays_info[info_id] = current_ray_info;
            array_of_rays_info[active_rays_count + info_id] = reflected_ray_info;
        }
    }
}


/*
  Note: This pass init start ray for each pixel, using position of viewer and transform matrix.
        Each ray will have normilized direction and inited by `viewer_position` started position.

        In this function for each pixel defined local vector of direction to it from viewer `viewer_position` and 
        with matrix multiplication in order to transform local coords to earth coords. we compute true 
        direction of each pixel from big image.
        Also we bind each ray to pixel using coords in order to compute shadings in recursive mode.
*/
__global__
void init_vewer_back_rays(recursion* array_of_rays_info, float_3* image, mat_3x3 transform_matrix, float_3 viewer_position, float z, uint32_t w, uint32_t h){
    uint32_t thread_step = blockDim.x * gridDim.x;
    uint32_t thread_start = threadIdx.x + blockIdx.x*blockDim.x;
    uint32_t active_rays_count = w * h;
    
    // TODO: compute this on CPU before this kernel launch
    float d_w = 2.0 / (w - 1.0);
    float d_h = 2.0 / (h - 1.0);
    float h_div_w = static_cast<float>(h) / static_cast<float>(w);

    for(uint32_t idx = thread_start; idx < active_rays_count; idx += thread_step){
        uint32_t i = idx % w;
        uint32_t j = idx / w;

        recursion active_ray_field;

        // Bind pixel to ray:
        active_ray_field.x = i; active_ray_field.y = j;
        active_ray_field.power = 1.0;
        active_ray_field.ray.from = viewer_position;

        // Define local coords of pixel. `pc` has coords (0, 0) here.
        float_3 pixel_coordinats = {-1.0f + d_w * i, (-1.0f + d_h * j) * h_div_w, z};

        // Transform local to earth coords:
        active_ray_field.ray.dir = norm(mult(transform_matrix, pixel_coordinats));

        array_of_rays_info[idx] = active_ray_field;
        image[idx] = {0.0, 0.0, 0.0}; // BLACK
    }
}


/*
  Note: In fact this function compute Average Pooling with filter and stride size
        equal to (sqrt, sqrt). This pass needed for Antialiasing.
*/
__global__
void ssaa(uint32_t* picture, const float_3* image, uint32_t w, uint32_t h, uint32_t sqrt_per_pixel){
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t thread_step_x = blockDim.x * gridDim.x;
    uint32_t thread_step_y = blockDim.y * gridDim.y;

    uint32_t big_w = w * sqrt_per_pixel;

    for(int32_t i = idx; i < h; i += thread_step_x){
        for(int32_t j = idy; j < w; j += thread_step_y){

            uint32_t thread_start_y = i * sqrt_per_pixel;
            uint32_t thread_start_x = j * sqrt_per_pixel;

            float_3 mean = {0.0, 0.0, 0.0};

            // Compute single pixel of picture as average value of window.
            for(int n = thread_start_y; n < thread_start_y + sqrt_per_pixel; ++n){
                for(int m = thread_start_x; m < thread_start_x + sqrt_per_pixel; ++m){
                    mean += image[n*big_w + m];
                }
            }
            mean /= static_cast<float>(sqrt_per_pixel*sqrt_per_pixel);

            // Write value to picture
            picture[i*w + j] = float3_to_uint32(clamp_pixel_color(mean));
        }
    }

}

#endif // __RAY_TRACING_CUH__