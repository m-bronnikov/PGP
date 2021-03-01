// Made by Max Bronnikov
#ifndef __CAMERA_CUH__
#define __CAMERA_CUH__

#include <cstdio>
#include <stdexcept>
#include <math.h>
#include "structures.cuh"


// Camera - generator of scene positions. returns basis of camera in earths coord while 
// update of position could be done. 
class Camera{
public:
    // initializer
    Camera(
        uint32_t frames, float angle,
        float rc0, float zc0, float fc0, float Acr, float Acz,
        float wcr, float wcz, float wcf, float pcr, float pcz,
        float rn0, float zn0, float fn0, float Anr, float Anz,
        float wnr, float wnz, float wnf, float pnr, float pnz
    ) :  
    count_frames(frames), view_z(1.0 / tan(angle * M_PI / 360.0)),
    time_step(2.0*M_PI/frames), current_time(-time_step),
    rc_0(rc0), zc_0(zc0), fc_0(fc0), Ac_r(Acr), Ac_z(Acz), 
    wc_r(wcr), wc_z(wcz), wc_f(wcf), pc_r(pcr), pc_z(pcz),
    rn_0(rn0), zn_0(zn0), fn_0(fn0), An_r(Anr), An_z(Anz), 
    wn_r(wnr), wn_z(wnz), wn_f(wnf), pn_r(pnr), pn_z(pnz){}

private:
    float_3 cilindric_to_decart(float r, float z, float f){
        return {r * cos(f), r * sin(f), z};
    }

    void compute_position(float t){
        float rc = rc_0 + Ac_r*sin(wc_r*t + pc_r);
        float zc = zc_0 + Ac_z*sin(wc_z*t + pc_z);
        float fc = fc_0 + wc_f*t;

        float rn = rn_0 + An_r*sin(wn_r*t + pn_r);
        float zn = zn_0 + An_z*sin(wn_z*t + pn_z);
        float fn = fn_0 + wn_f*t;

        pc = cilindric_to_decart(rc, zc, fc);
        float_3 pn = cilindric_to_decart(rn, zn, fn);

        float_3 vz = norm(pn - pc);
        float_3 vx = norm(cross(vz, {0.0, 0.0, 1.0}));
        float_3 vy = norm(cross(vx, vz));

        view_matrix = {vz, vx, vy};
    }

public:
    bool update_position(){
        current_time += time_step;
        compute_position(current_time);

        return current_time < 2.0 * M_PI;
    }

    mat_3x3 get_frame_basis(){
        return view_matrix;
    }

    float_3 get_camera_position(){
        return pc;
    }

    uint32_t count_of_frames(){
        return count_frames;
    }

    float get_z_coord(){
        return view_z;
    }

private:
    mat_3x3 view_matrix;
    float_3 pc;
    float view_z;

    uint32_t count_frames;

    float time_step;
    float current_time;

    float rc_0;
    float zc_0;
    float fc_0;
    float Ac_r;
    float Ac_z;
    float wc_r;
    float wc_z;
    float wc_f;
    float pc_r;
    float pc_z;

    float rn_0;
    float zn_0;
    float fn_0;
    float An_r;
    float An_z;
    float wn_r;
    float wn_z;
    float wn_f;
    float pn_r;
    float pn_z;
};


#endif // __CAMERA_CUH__