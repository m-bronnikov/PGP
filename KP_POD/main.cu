// Made by Max Bronnikov
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include "figures.cuh"
#include "structures.cuh"
#include "ray_tracing.cuh"
#include "scene.cuh"

using namespace std;


int main(){

    // Need to read
    uint32_t frames;

    string path_modifier;

    uint32_t width, height;
    float view_angle;

    float rc_0, zc_0, fc_0, Ac_r, Ac_z, wc_r, wc_z, wc_f, pc_r, pc_z;
    float rn_0, zn_0, fn_0, An_r, An_z, wn_r, wn_z, wn_f, pn_r, pn_z;

    float_3 center, color;
    float radius;
    float reflection, refraction;
    uint32_t line_lights;

    float_3 floor_A, floor_B, floor_C, floor_D;
    string texture_path;
    float_3 texture_color;
    float texture_refl;

    uint32_t light_num;

    uint32_t recursion_depth, sqrt_per_pixel;

    // read:
    // 1.
    cin >> frames;

    // 2.
    cin >> path_modifier;

    // 3.
    cin >> width >> height;
    cin >> view_angle;

    // 4.
    cin >> rc_0 >> zc_0 >> fc_0 >> Ac_r >> Ac_z >> wc_r >> wc_z >> wc_f >> pc_r >> pc_z;
    cin >> rn_0 >> zn_0 >> fn_0 >> An_r >> An_z >> wn_r >> wn_z >> wn_f >> pn_r >> pn_z;
    // create main objects
    FileWriter writter(width, height, path_modifier);
    Camera camera(
        frames, view_angle, 
        rc_0, zc_0, fc_0, Ac_r, Ac_z, wc_r, wc_z, wc_f, pc_r, pc_z,
        rn_0, zn_0, fn_0, An_r, An_z, wn_r, wn_z, wn_f, pn_r, pn_z
    );

    Scene scene(camera, writter);

    // 5.
    // TODO: change figures
    // dodecaedr:
    cin >> center.x >> center.y >> center.z;
    cin >> color.x >> color.y >> color.z;
    cin >> radius;
    cin >> reflection >> refraction;
    cin >> line_lights; 
    material fig_glass = {color, 1.0, reflection, refraction}; // set diffusion as 1 by default
    scene.add_figure(Dodecaedr(center, fig_glass, radius, line_lights));

    // dodecaedr:
    cin >> center.x >> center.y >> center.z;
    cin >> color.x >> color.y >> color.z;
    cin >> radius;
    cin >> reflection >> refraction;
    cin >> line_lights; 
    fig_glass = {color, 1.0, reflection, refraction}; // set diffusion as 1 by default
    scene.add_figure(Dodecaedr(center, fig_glass, radius, line_lights));

    // dodecaedr:
    cin >> center.x >> center.y >> center.z;
    cin >> color.x >> color.y >> color.z;
    cin >> radius;
    cin >> reflection >> refraction;
    cin >> line_lights; 
    fig_glass = {color, 1.0, reflection, refraction}; // set diffusion as 1 by default
    scene.add_figure(Dodecaedr(center, fig_glass, radius, line_lights));

    // 6.
    cin >> floor_A.x >> floor_A.y >> floor_A.z;
    cin >> floor_B.x >> floor_B.y >> floor_B.z;
    cin >> floor_C.x >> floor_C.y >> floor_C.z;
    cin >> floor_D.x >> floor_D.y >> floor_D.z;
    cin >> texture_path;
    cin >> texture_color.x >> texture_color.y >> texture_color.z;
    cin >> texture_refl;
    // material text_mat = {texture_color, 1.0, texture_refl, 0.0}; // setd diffusion as 1.0

    // 7.
    cin >> light_num;
    for(uint8_t i = 0; i < light_num; ++i){
        float_3 light_coord, light_color;
        cin >> light_coord.x >> light_coord.y >> light_coord.z;
        cin >> light_color.x >> light_color.y >> light_color.z;

        scene.add_light({light_coord, light_color, 1.0}); // set power as 1 by default
    }

    // 8.
    cin >> recursion_depth >> sqrt_per_pixel;

    // set window before render:
    scene.set_window(width, height, sqrt_per_pixel);

    // launch render ;)
    scene.gpu_render_scene(recursion_depth);

    cout << "Program successfully ends!" << endl;

    return 0;
}