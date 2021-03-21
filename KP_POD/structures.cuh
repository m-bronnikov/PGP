// Made by Max Bronnikov
#ifndef __STRUCTURES_CUH__
#define __STRUCTURES_CUH__
#include <stdexcept>
#include <string>

/* Note: Vector with 3 elem's

   It's a base structure of this project in all geometric operations and structures.
*/
struct float_3
{   
    float x;
    float y;
    float z;
};

/*
    Note: Matrix 3x3 with following form:

                / a.x b.x c.x \
               |  a.y b.y c.y |
                \ a.z b.z c.z /

    It may be usefull for definition of transformation matrix between basises.
*/
struct mat_3x3{
    float_3 a;
    float_3 b;
    float_3 c;
};

/*
    Note: Ray

    We define ray as coordinat of start position(`from`) with vector of direction(`dir`).

    This class named `back_ray` because we pass rays in backward direction from viewer to light sources.
*/
struct back_ray
{
    float_3 from;
    float_3 dir; // should be normilized
};

/* Note: Triangle's properties of material

        This structure describes triangle material from scene. 
        Main properties:
            1. `color` - color of triangle
            2. `diffusion` - for Phong(Lambert) shading computation
            3. `reflection` - power of reflected ray (also may be used as coeff of specular part of phong shading).
            4. `refraction` - coeff of refraction of material.
*/ 
struct material{
    float_3 color;
    float diffussion;
    float reflection;
    float refraction;
};

/* Note: Triangle defineition structure

   At first revision this structure has `float_3 normal` attribute, but
   it was decided to compute normal in runtime because it's not memory friendly to store 
   this 12 bites for rare usage.

   Also for memory economy I store only material id from another propertie table 
   instead storing color, reflection, refraction and defusion(optional) for each triangle of scene.
*/
struct triangle
{
    float_3 a;
    float_3 b;
    float_3 c;
    uint32_t mat_id;
};

/*
    Note: Light source properties.

    In this work we use simple light point sources.
    Each source has coordinats on scene, color of shine and power of radiocity.
*/
struct light_point
{
    float_3 coord;
    float_3 color;
    float power;
};

/*
    Note: Computation data of rays.

    This structure - extentended description of each ray, needed for recursive computation.
    Main properties:
    1. `ray` 
    2. `power` - in recursion power of passed ray reducing with each intersection. 
        For our ray it's define factor of influence to overall pixel color from image.
    3-4. `x` and `y` - image coordinates of binded to this ray pixels.
*/
struct recursion{
    back_ray ray;
    float power;
    // Q. May be better to use uint16_t for mem economy?
    uint32_t x;
    uint32_t y;
};

/* Note: Texture properties.

  // TODO reafactor this
*/
struct textures{
    float_3 A;
    float_3 B;
    float_3 C;
    float_3 D;
    float_3 color;
};

// Hash function for material's
struct material_functor
{
    size_t operator()(const material& mat) const{
        return 103245*mat.color.x + 87599*mat.color.y + 
            73245*mat.color.z + 23412*mat.diffussion + 
            3564653*mat.refraction + 453452*mat.reflection;
    }
};

// For collision resolving
bool operator==(const material& lhs, const material& rhs){
    return lhs.color.x == rhs.color.x && lhs.color.y == rhs.color.y && lhs.color.z == rhs.color.z &&
    lhs.diffussion == rhs.diffussion && lhs.reflection == rhs.reflection && lhs.refraction == rhs.refraction;
}

/*
    Note: Set of operators and base geometric functions for working with `float_3`
*/
__host__ __device__
float_3 operator*(const float_3& lhs, float rhs){
    return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}

__host__ __device__
float_3 operator*(const float_3& lhs, const float_3& rhs){
    return {lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z};
}

__host__ __device__
float_3 operator/(const float_3& lhs, float rhs){
    return {lhs.x/rhs, lhs.y/rhs, lhs.z/rhs};
}

__host__ __device__
float_3 operator+(const float_3& lhs, const float_3& rhs){
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

__host__ __device__
float_3 operator-(const float_3& lhs, const float_3& rhs){
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

__host__ __device__
float_3 operator-(const float_3& lhs){
    return {-lhs.x, -lhs.y, -lhs.z};
}

__host__ __device__
float_3& operator+=(float_3& lhs, const float_3& rhs){
    lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z;
    return lhs;
}

__host__ __device__
float_3& operator-=(float_3& lhs, const float_3& rhs){
    lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z;
    return lhs;
}

__host__ __device__
float_3& operator/=(float_3& lhs, float rhs){
    lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs;
    return lhs;
}

__host__ __device__
float_3& operator*=(float_3& lhs, float rhs){
    lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs;
    return lhs;
}

__host__ __device__
float_3& operator*=(float_3& lhs, const float_3& rhs){
    lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z;
    return lhs;
}

__host__ __device__
float dot(const float_3& lhs, const float_3& rhs){
    return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}

__host__ __device__
float abs(const float_3& lhs){
    return sqrt(dot(lhs, lhs));
}

// Q. may be better to return as int?
__host__ __device__
float sign(const float lhs){
    if(lhs == 0.0f){
        return 0.0f;
    }

    return lhs > 0.0f ? 1.0f : -1.0f;
}

__host__ __device__
float_3 norm(const float_3& lhs){
    return lhs / abs(lhs);
}

__host__ __device__
float_3 cross(const float_3& lhs, const float_3& rhs){
    return {lhs.y*rhs.z - lhs.z*rhs.y, lhs.z*rhs.x - lhs.x*rhs.z, lhs.x*rhs.y - lhs.y*rhs.x};
}

__host__ __device__
float_3 mult(const mat_3x3& lhs, const float_3& rhs){
    return {lhs.a.x * rhs.x + lhs.b.x * rhs.y + lhs.c.x * rhs.z,
        lhs.a.y * rhs.x + lhs.b.y * rhs.y + lhs.c.y * rhs.z,
        lhs.a.z * rhs.x + lhs.b.z * rhs.y + lhs.c.z * rhs.z};
}

/* Note: Define normilized normal of triangle

        For triangle (A B C):
                         A
                         /\
                     a2 /  \ a1
                       /    \
                     |/_    _\|
                     ----------
                    B         C
    It's easy to define normal:
        1. Find vector a1 = C - A
        2. Find vector a2 = B - A
        3. Define normal = cross(a1, a2) because cross product 
           returns ort vector for a1 and a2 both.
        4(optional). normilize obtained vector: normal = norm(normal) 
*/
__host__ __device__
float_3 normal_of_triangle(const triangle& trig){
    return norm(cross(trig.c - trig.a, trig.b - trig.a));
}

__host__ __device__
uint32_t float3_to_uint32(float_3 color){
    uint32_t result = 0;

    result |= static_cast<uint32_t>(round(255.0 * color.x)) << 0; //RED
    result |= static_cast<uint32_t>(round(255.0 * color.y)) << 8; //GREEN
    result |= static_cast<uint32_t>(round(255.0 * color.z)) << 16; //BLUE

    return result;
}

/* Note: atomicAdd for float_3 datatype

    Real atomicAdd must return value before operation, but we don't need it.
*/
__device__
void atomicAdd(float_3* adress, const float_3& val){
    atomicAdd(&adress->x, val.x);
    atomicAdd(&adress->y, val.y);
    atomicAdd(&adress->z, val.z);
}

// Note: Error handler throws runtime_error if cuda erros occured
void throw_on_cuda_error(const cudaError_t& code)
{
  if(code != cudaSuccess)
  {
    std::string err_str = "CUDA Error: ";
    err_str += cudaGetErrorString(code);
    throw std::runtime_error(err_str);
  }
}

// TODO Optimize this
/* Note: Clamps value to correct range.
        Values from each channel of pixel should be in [0.0, 1.0]
         If value > 1.0 => clamp this to 1.0!
*/
__device__ __host__
float_3 clamp_pixel_color(const float_3& pixel){
    return {min(pixel.x, 1.0f), min(pixel.y, 1.0f), min(pixel.z, 1.0f)};
}

#endif // __STRUCTURES_CUH__