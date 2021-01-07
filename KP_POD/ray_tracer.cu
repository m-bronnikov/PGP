#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include "figures.cuh"
#include "structures.cuh"
#include "ray_tracing.cuh"

using namespace std;

// Settings:
#define SQRT_RAYS_PER_PIXEL 2


/*
class render_scene{
public:
    render_scene();
private:
    vector<triangle> render_triangles;
    string path_to_dir;
    uint32_t sqrt_rays_per_pixel;
};
*/



int main(){
    vector<triangle> triangles;
    vector<light_point> lights;
    dodecaedr fig_1(10.0, 5);
    fig_1.render_figure(triangles, lights);

    cout << "here " << triangles.size() << endl;

    return 0;
}