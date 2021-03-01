// Made by Max Bronnikov
#ifndef __RAY_TRACING_CUH__
#define __RAY_TRACING_CUH__

#include "structures.cuh"

#define THREADS 256
#define BLOCKS 256

#define MAX_FLOAT 1e+32
#define EPSILON 1e-10
#define SPEC_POW 32

struct intersection
{
    float distance;
    uint32_t id; // id of triangle
};


//  texture<uint32_t, 2, cudaReadModeElementType> g_text;

// baricentric test
__device__
float triangle_intersected(const back_ray* ray, triangle* c_trig){
    float_3 e1 = c_trig->b - c_trig->a;
    float_3 e2 = c_trig->c - c_trig->a;
    float_3 p = cross(ray->dir, e2);
    float div_n = dot(p, e1);

    if (d_abs(div_n) < EPSILON){
        return -1.0f;
    }

    float_3 t = ray->start - c_trig->a;

    float u = dot(p, t) / div_n;

    float_3 q = cross(t, e1);

    float v = dot(q, ray->dir) / div_n;

    float ts = dot(q, e2) / div_n; 	

    // if/else 3 times may slow our program, beter do extra computations below and run if/else at ones
    if(u >= 0.0 && v >= 0.0 && v + u < 1.0){
        return ts;
    }

    return -1.0;
}

__device__
void search_triangle_intersection(
    intersection* ans, const back_ray* ray, const triangle* trigs, uint32_t trig_count
){
    for(uint32_t i = 0; i < trig_count; ++i){
        triangle c_trig = trigs[i];

		float ts = triangle_intersected(ray, &c_trig); 	

		if(ts >= 0.0 && ts < ans->distance){
			ans->distance = ts;
			ans->id = i;
        }
    }
}

// this function trace ray and computes power of ray passed throught triangles
__device__
float_3 compute_ray_power(
    const back_ray* ray, const material* mats, const triangle* trigs, 
    uint32_t trig_count, uint32_t r_trig, float l_dist
){
    float_3 power = {1.0, 1.0, 1.0};
    for(uint32_t i = 0; i < trig_count; ++i){
        triangle c_trig = trigs[i];

		float ts = triangle_intersected(ray, &c_trig); 	

        // r_trig check needs for avoid intersection with source triangle
		if(ts >= 0.0 && ts < l_dist && i != r_trig){
            // if ray intersected beetwen light and triangle => correct light power
            material trig_mat = mats[c_trig.mat_id];
            power *= trig_mat.color;
            power *= trig_mat.refraction;
        }
    }

    return power;
}

__device__
float_3 reflect(float_3 ray, float_3 normal){
    float_3 ref = normal;
    ref *= 2.0f * dot(normal, ray);
    ref -= ray;
    return ref;
}

// compute coeff for phong shading
__device__
float phong_shading(float_3 normal, float_3 v_r, float_3 l_r, material t_mat){
    float shading = 0.0f;
    
    // diffusion
    shading += t_mat.diffussion * dot(normal, l_r);
    
    // specular
    shading += t_mat.reflection * max(0.0, pow(dot(reflect(l_r, normal), v_r), SPEC_POW));

    return shading;
}

__global__
void gpu_ray_trace(
    recursion* rec_arr, uint32_t rec_count,
    float_3* img, material* mats,
    triangle* trigs, uint32_t trig_count,
    light_point* lights, uint32_t light_count,
    uint32_t w, uint32_t h
){
    uint32_t step = blockDim.x * gridDim.x;
    uint32_t start = threadIdx.x + blockIdx.x*blockDim.x;
    
    for(uint32_t r_id = start; r_id < rec_count; r_id += step){
        // main computations for single ray
        intersection travers; 
        travers.distance = MAX_FLOAT;
        travers.id = trig_count;
        recursion rec_field = rec_arr[r_id];

        // found intersection:
        search_triangle_intersection(&travers, &rec_field.ray, trigs, trig_count);
        if(travers.id == trig_count){
            continue;
        }

        float_3 normal;
        material t_mat;
        {
            triangle trig = trigs[travers.id];
            t_mat = mats[trig.mat_id];
            normal = trig.normal;
            normal *= dot(normal, rec_field.ray.dir) ? -1.0f : 1.0f; // correct normal
        }

        float_3 travers_coord = rec_field.ray.start + rec_field.ray.dir*travers.distance;

        // compute shading for each light source
        float_3 sum_of_lights = {0.0, 0.0, 0.0};
        for(uint32_t l_id = 0; l_id < light_count; ++l_id){
            // get light_source and create ray to this point
            light_point shine = lights[l_id];
            float_3 vec_to_light = shine.coord - travers_coord;
            back_ray light_ray = {travers_coord, norm(vec_to_light)};

            // if path from shine to intersection doesnt exist => continue
            if(dot(normal, light_ray.dir) < EPSILON){
                continue;
            }

            // trace ray to point and compute solar power of source
            float_3 light_color = shine.color;
            light_color *= compute_ray_power(
                &light_ray, mats, trigs, trig_count, travers.id, abs(vec_to_light)
            );
            light_color *= shine.power;

            light_color *= phong_shading(normal, -rec_field.ray.dir, light_ray.dir, t_mat);

            sum_of_lights += light_color;
        }
        // store result to global mem
        img[rec_field.y * w + rec_field.x] += sum_of_lights;

        // search new rays
        {
            // reflection
            recursion refl_ray = rec_field;
            refl_ray.power *= t_mat.reflection;
            refl_ray.ray.start = travers_coord;
            refl_ray.ray.dir = norm(reflect(-rec_field.ray.dir, normal));

            // refraction
            rec_field.ray.start = travers_coord;
            rec_field.power *= t_mat.refraction;

            // push back to memory
            rec_arr[r_id] = rec_field;
            rec_arr[rec_count + r_id] = refl_ray;
        }
    }
}

__global__
void init_vewer_back_rays(recursion* recs, float_3* clrs, mat_3x3 v_mat, float_3 pc, float z, uint32_t w, uint32_t h){
    uint32_t step = blockDim.x * gridDim.x;
    uint32_t start = threadIdx.x + blockIdx.x*blockDim.x;
    uint32_t size = w * h;
    
    // TODO: compute this on CPU before this kernel launch
    float d_w = 2.0 / (w - 1.0);
    float d_h = 2.0 / (h - 1.0);
    float h_div_w = h / w;

    // init rays with black color and direction and power of pass
    for(uint32_t idx = start; idx < size; idx += step){
        uint32_t i = idx % w;
        uint32_t j = idx / h;

        recursion r_field;
        r_field.x = i; r_field.y = j;
        r_field.power = 1.0;
        r_field.ray.start = pc;
        r_field.ray.dir =  norm(mult(v_mat, {-1.0f + d_w * i, (-1.0f + d_h * j) * h_div_w, z}));

        recs[idx] = r_field;
        clrs[idx] = {0.0, 0.0, 0.0};
    }
}

#endif // __RAY_TRACING_CUH__