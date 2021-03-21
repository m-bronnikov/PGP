#include <iostream>
#include <stdexcept>
#include <string>
#include <math.h>


struct float_3{
    float x;
    float y;
    float z;
};

struct triangle{
    float_3 a;
    float_3 b;
    float_3 c;
};

float dot(float_3 a, float_3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

float_3 diff(float_3 a, float_3 b){
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float_3 norm(float_3 v) {
	float l = sqrt(dot(v, v));
	return { v.x / l, v.y / l, v.z / l };
}

float_3 prod(float_3 a, float_3 b) {
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

float_3 cross(const float_3& lhs, const float_3& rhs){
    return {lhs.y*rhs.z - lhs.z*rhs.y, lhs.z*rhs.x - lhs.x*rhs.z, lhs.x*rhs.y - lhs.y*rhs.x};
}


using namespace std;

int main(){
    float_3 light_pos = {10, 10, 10};
    float_3 normal = {0, 0, 0};
    float_3 dir = diff({0, 0, 0}, {-1, 0, 1});
    triangle trig = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0}
    };

    //search normal
	{
		float_3 a1 = diff(trig.a, trig.b);
		float_3 a2 = diff(trig.a, trig.c);

        cout << "a1: " << a1.x << " " << a1.y << " " << a1.z << endl;
        cout << "a2: " << a2.x << " " << a2.y << " " << a2.z << endl;

		normal = cross(a1, a2);

		// correct noraml
		if(dot(dir, normal) > 0.0){
			normal.x *= -1.0;
			normal.y *= -1.0;
			normal.z *= -1.0;
		}
	}

    cout << "dir: " << dir.x << " " << dir.y << " " << dir.z << endl;
    cout << "normal: " << normal.x << " " << normal.y << " " << normal.z << endl;


    cout << "Dot:" << dot(norm(light_pos), normal) << endl;

    return 0;
}