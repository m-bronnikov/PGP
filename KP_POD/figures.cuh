#ifndef __FIGURES_CUH__
#define __FIGURES_CUH__

#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include "structures.cuh"

using namespace std;

class figure_3d{
protected:
    using point = float_3;
    using line = pair<uint8_t, uint8_t>;
    using edge = vector<uint8_t>;

public:
    void render_figure(vector<triangle>& triangles, vector<light_point>& lights) const{
        set_lights(lights);
        split_line_triangles(triangles);
        split_edge_triangles(triangles);
    }

protected:
    void generate_figure(float radius_o){
        generate_vertexes();
        scale_vertexes(radius_o);
    }

    void scale_vertexes(float radius){
        for(auto& v : vertexes){
            v *= radius;
        }
    }

    // this function generates vertexes of figure with R = 1
    virtual void generate_vertexes() = 0;

    // this function generate light balls throuth figure angle
    void set_lights(vector<light_point>& lights) const{
        for(auto& line_idx : lines){
            // 1) compute vector of transform:
            float_3 transform =  norm(vertexes[line_idx.first] + vertexes[line_idx.second]);
            transform *= light_radius / tan((M_PI - angle_between_edges) / 2.0) - light_radius; // len from line to center of light
            
            // 2) generate light balls
            float_3 start = vertexes[line_idx.first];
            for(uint8_t i = 1; i <= lights_per_line; ++i){
                float_3 add = (vertexes[line_idx.second] - vertexes[line_idx.first]);
                add *= ((float) i) / ((float) (lights_per_line + 1));
                // center of light ball, radius of ball, color of light, power of light
                lights.push_back({start + add + transform, light_radius, light_color, light_power});
            }

        }
    }

    virtual void split_edge_triangles(vector<triangle>& triangles) const = 0;

    void split_line_triangles(vector<triangle>& triangles) const{
        for(auto& line_idx : lines){
            // work with angle here.   
            // 1) compute vector of transform:
            float_3 transform =  norm(vertexes[line_idx.first] + vertexes[line_idx.second]);
            float_3 gap = norm(cross(transform, vertexes[line_idx.second] - vertexes[line_idx.first]));

            // lens of transform and gap
            gap *= light_radius;
            transform *= light_radius / tan((M_PI - angle_between_edges) / 2.0) - light_radius; // len from line to center of light
            
            // 2) generate 2 triangles for agle
            triangles.push_back({
                vertexes[line_idx.first] + transform + gap, 
                vertexes[line_idx.first] + transform - gap, 
                vertexes[line_idx.second] + transform - gap, 
            });
            triangles.push_back({
                vertexes[line_idx.second] + transform + gap, 
                vertexes[line_idx.second] + transform - gap, 
                vertexes[line_idx.first] + transform + gap, 
            });
        }
    }

protected:
    float light_radius = 0.005;
    float_3 light_color = {1.0, 1.0, 1.0};
    float light_power = 0.1;
    uint8_t lights_per_line;

    vector<line> lines;
    vector<edge> edges;

    float scale_radius;
    float angle_between_edges;
    float_3 position;

    vector<point> vertexes;
};



class dodecaedr : public figure_3d{
public:
    dodecaedr(float radius_o, uint8_t line_light){
        // set dodecaedr params:
        lights_per_line = line_light;
        angle_between_edges = 2.034443043;

        set_lines_edges();
                
        // generate dodecaedr
        generate_figure(radius_o);
    }

private:
    // this function generates vertexes of dodecaedr with R = 1
    void generate_vertexes() final{
        vertexes.resize(20);

        const vector<uint8_t> G0 = {0, 2, 4, 6, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 3, 5, 7, 9};
        const vector<uint8_t> G1 = {10, 10, 10, 10, 10, 2, 2, 4, 4, 6, 6, 8, 8, 0, 0, 11, 11, 11, 11, 11};
        const vector<uint8_t> G2 = {2, 4, 6, 8, 0, 1, 3, 3, 5, 5, 7, 7, 9, 9, 1, 9, 1, 3, 5, 7};

        vector<point> ico_vertxs;
        generate_icosaedr(ico_vertxs);

        // coords of dodecaedr:
        for(uint8_t i = 0; i < 20; ++i){
            vertexes[i] = ico_vertxs[G0[i]] + ico_vertxs[G1[i]] + ico_vertxs[G2[i]];
            vertexes[i] /= 3.0;
        }
    }

    void split_edge_triangles(vector<triangle>& triangles) const final{
        for(auto& edge_idxs : edges){
            float_3 transform = {0.0, 0.0, 0.0};
            for(auto v_idx : edge_idxs){
                transform += vertexes[v_idx];
            }

            transform = norm(transform);
            float_3 normal = transform;
            transform *= light_radius / sin((M_PI - angle_between_edges) / 2.0);

            // add triangles
            triangles.push_back({
                vertexes[edge_idxs[0]] + transform, vertexes[edge_idxs[1]] + transform, vertexes[edge_idxs[2]] + transform, normal
            });
            triangles.push_back({
                vertexes[edge_idxs[2]] + transform, vertexes[edge_idxs[3]] + transform, vertexes[edge_idxs[4]] + transform, normal
            });
            triangles.push_back({
                vertexes[edge_idxs[4]] + transform, vertexes[edge_idxs[2]] + transform, vertexes[edge_idxs[0]] + transform, normal
            });
        }
    }

private:
    // this function generates vertexes of icosaedr with Ro = 1.118
    void generate_icosaedr(vector<point>& ico_vertxs) const{
        ico_vertxs.resize(12);
        float h = 0.5;
        const float Ro = 1.118034;
        // coords of icosaedr
        for(uint8_t i = 0; i < 10; ++i){
            ico_vertxs[i] = {
                static_cast<float>(cos(i * M_PI / 5.0f)), 
                h,
                static_cast<float>(sin(i * M_PI / 5.0f))
            };
            h *= -1.0;
        }
        ico_vertxs[10] = {0, Ro, 0};
        ico_vertxs[11] = {0, -Ro, 0};   
    }

    void set_lines_edges(){
        lines = {
            {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 5}, {5, 6},
            {6, 7}, {7, 1}, {8, 7}, {9, 8}, {2, 9}, {9, 10}, {10, 11},
            {11, 3}, {12, 11}, {13, 12}, {4, 13}, {13, 14}, {14, 5}, {15, 14},
            {16, 15}, {6, 16}, {16, 17}, {17, 8}, {18, 17}, {10, 18}, {18, 19},
            {19, 12}, {15, 19}
        };

        vector<edge> edges = {
            {0, 1, 2, 3, 4}, {15, 16, 17, 18, 19},
            {6, 7, 8, 17, 16}, {6, 5, 14, 15, 16},
            {12, 13, 14, 15, 19}, {18, 19, 12, 11, 10},
            {10, 9, 8, 17, 18}, {0, 1, 7, 6, 5},
            {4, 0, 5, 14, 13}, {13, 4, 3, 11, 12},
            {11, 3, 2, 9, 10}, {8, 9, 2, 1, 7}
        };
    }
};


#endif // __FIGURES_CUH__