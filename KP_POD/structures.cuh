// Made by Max Bronnikov
#ifndef __STRUCTURES_CUH__
#define __STRUCTURES_CUH__
#include <stdexcept>
#include <string>

struct float_3
{   
    float x;
    float y;
    float z;
};

struct mat_3x3{
    float_3 a;
    float_3 b;
    float_3 c;
};

struct back_ray
{
    float_3 start;
    float_3 dir;
};

struct material{
    float_3 color;
    float diffussion;
    float reflection;
    float refraction;
};

struct triangle
{
    float_3 a;
    float_3 b;
    float_3 c;
    float_3 normal; // for Phong lightning
    uint32_t mat_id;
};


struct light_point
{
    float_3 coord;
    float_3 color;
    float power;
};

struct recursion{
    back_ray ray;
    float power;
    // Q. May be better to use uint16_t for mem economy
    uint32_t x;
    uint32_t y;
};

struct textures{
    float_3 A;
    float_3 B;
    float_3 C;
    float_3 D;
    float_3 color;
};

struct material_functor
{
    size_t operator()(const material& mat) const{
        return 103245.0*(mat.color.x + mat.color.y + mat.color.z + mat.diffussion + mat.refraction + mat.reflection);
    }
};

bool operator==(const material& lhs, const material& rhs){
    return lhs.color.x == rhs.color.x && lhs.color.y == rhs.color.y && lhs.color.z == rhs.color.z &&
    lhs.diffussion == rhs.diffussion && lhs.reflection == rhs.reflection && lhs.refraction == rhs.refraction;
}

__host__ __device__
float_3 operator*(const float_3& lhs, const float rhs){
    return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
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

__host__ __device__
float d_max(const float lhs, const float rhs){
    return lhs > rhs ? lhs : rhs;
}

__host__ __device__
float d_abs(const float lhs){
    return lhs < 0.0f ? -lhs : lhs; 
}

__host__ __device__
float_3 norm(const float_3& lhs){
    float_3 ans = lhs;
    ans /= abs(ans);
    return ans;
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

__host__ __device__
uint32_t float3_to_uint32(float_3 color){
    uint32_t result = 0;

    result |= ((uint32_t) (255.0 * color.x)) << 0; //RED
    result |= ((uint32_t) (255.0 * color.y)) << 8; //GREEN
    result |= ((uint32_t) (255.0 * color.z)) << 16; //BLUE

    return result;
}

// Error handler
void throw_on_cuda_error(const cudaError_t& code)
{
  if(code != cudaSuccess)
  {
    std::string err_str = "CUDA Error: ";
    err_str += cudaGetErrorString(code);
    throw std::runtime_error(err_str);
  }
}

void normalized_to_rgb(const float_3* colors, uint32_t* rgb, uint32_t size){
    for(uint32_t i = 0; i < size; ++i){
        rgb[i] = float3_to_uint32(colors[i]);
    }
}

__global__
void cuda_normalized_to_rgb(const float_3* colors, uint32_t* rgb, uint32_t size){
    uint32_t idx = blockDim.x * blockIdx.x +  threadIdx.x;
    uint32_t step = blockDim.x * gridDim.x;

    for(uint32_t i = idx; i < size; i += step){
        rgb[i] = float3_to_uint32(colors[i]);
    }
}

#endif // __STRUCTURES_CUH__