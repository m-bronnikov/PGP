#ifndef __STRUCTURES_CUH__
#define __STRUCTURES_CUH__


struct float_3
{   
    float x;
    float y;
    float z;
};

struct back_ray
{
    float_3 start;
    float_3 dir;
};

struct triangle
{
    float_3 A;
    float_3 B;
    float_3 C;
    float_3 normal; // for Phong lightning
};

struct light_point
{
    float_3 coord;
    float radius;
    float_3 color;
    float light_power;
};


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
float dot(const float_3& lhs, const float_3& rhs){
    return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}

__host__ __device__
float abs(const float_3& lhs){
    return sqrt(dot(lhs, lhs));
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


#endif // __STRUCTURES_CUH__