// Made by Max Bronnikov
#ifndef __FIGURES_CUH__
#define __FIGURES_CUH__

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include "structures.cuh"
#include "material_table.cuh"

using namespace std;

#define DIST_EPS 1e-8


class Figure3d{
protected:
    using point = float_3;
    using line = pair<uint8_t, uint8_t>;
    using edge = vector<uint8_t>;

public:
    Figure3d(const float_3& center_pos, const material& glass_mat) : center_coords(center_pos), trig_material(glass_mat){}

    void render_figure(vector<triangle>& triangles) const{
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

    triangle triangle_to_earth_coords(const triangle& single_triangle) const{
        triangle result = single_triangle;
        
        result.a += center_coords; 
        result.b += center_coords; 
        result.c += center_coords;
        
        return result;
    }

protected:
    // this function generates vertexes of figure with R = 1
    virtual void generate_vertexes() = 0;

    // this function split each edge to triangles
    virtual void split_edge_triangles(vector<triangle>& triangles) const = 0;

protected:
    // this function split each angle to triangles
    void split_line_triangles(vector<triangle>& triangles) const{
        uint32_t material_id = MaterialTable().get_material_id(angleline_material);

        for(auto& line_idx : lines){
            // work with angle here.   
            // 1) compute vector of transform:
            float_3 transform = norm(vertexes[line_idx.first] + vertexes[line_idx.second]);
            float_3 gap = norm(cross(transform, vertexes[line_idx.second] - vertexes[line_idx.first]));
            float_3 normal = transform;
            // lens of transform and gap
            gap *= lamp_radius;

            transform *= lamp_radius / tan((M_PI - angle_between_edges) / 2.0); 

            
            // 2) generate 2 triangles for agle
            triangles.push_back(triangle_to_earth_coords({
                vertexes[line_idx.first] + transform + gap, 
                vertexes[line_idx.first] + transform - gap, 
                vertexes[line_idx.second] + transform - gap, 
                normal,
                material_id
            }));

            triangles.push_back(triangle_to_earth_coords({
                vertexes[line_idx.second] + transform + gap, 
                vertexes[line_idx.second] + transform - gap, 
                vertexes[line_idx.first] + transform + gap, 
                normal,
                material_id
            }));

            // 3) generate lamp balls
            material_id = MaterialTable().get_material_id(diod_material);
            transform *= (1 - DIST_EPS);
            float_3 start = vertexes[line_idx.first];
            for(uint8_t i = 1; i <= lamps_per_line; ++i){
                float_3 add = (vertexes[line_idx.second] - vertexes[line_idx.first]);
                float_3 up = norm(add);

                up *= lamp_radius;
                add *= ((float) i) / ((float) (lamps_per_line + 1));

                // center of lamp ball, radius of ball, color of lamp, power of lamp

                float_3 center_coords = start + add + transform;

                triangles.push_back(triangle_to_earth_coords({
                    center_coords + up, 
                    center_coords - gap, 
                    center_coords + gap, 
                    normal,
                    material_id
                }));
            }
        }
    }

protected:
    uint8_t lamps_per_line;
    const float lamp_radius = 0.001;
    const material angleline_material = {{0, 0, 0}, 0.75, 0, 0};
    const material diod_material = {{0.85, 0.85, 0.85}, 0.8, 0.9, 0};

    material trig_material;

    vector<line> lines;
    vector<edge> edges;

    float angle_between_edges;
    const float_3 center_coords;

    vector<point> vertexes;
};



class Dodecaedr : public Figure3d{
public:
    Dodecaedr(const float_3& position, const material& glass_mat, float radius_o, uint8_t line_lamps) 
    : Figure3d(position, glass_mat){
        // set Dodecaedr params:
        lamps_per_line = line_lamps;
        angle_between_edges = 2.034443043;

        set_lines_edges();
                
        // generate Dodecaedr
        generate_figure(radius_o);
    }

private:
    // this function generates vertexes of Dodecaedr with R = 1
    void generate_vertexes() final{
        vertexes.resize(20);

        const vector<uint8_t> G0 = {0, 2, 4, 6, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 3, 5, 7, 9};
        const vector<uint8_t> G1 = {10, 10, 10, 10, 10, 2, 2, 4, 4, 6, 6, 8, 8, 0, 0, 11, 11, 11, 11, 11};
        const vector<uint8_t> G2 = {2, 4, 6, 8, 0, 1, 3, 3, 5, 5, 7, 7, 9, 9, 1, 9, 1, 3, 5, 7};

        vector<point> ico_vertxs;
        generate_icosaedr(ico_vertxs);

        // coords of Dodecaedr:
        for(uint8_t i = 0; i < 20; ++i){
            vertexes[i] = ico_vertxs[G0[i]] + ico_vertxs[G1[i]] + ico_vertxs[G2[i]];
            vertexes[i] /= 3.0;
        }
    }

    void split_edge_triangles(vector<triangle>& triangles) const final{
        uint32_t material_id = MaterialTable().get_material_id(trig_material);

        for(auto& edge_idxs : edges){
            float_3 transform = {0.0, 0.0, 0.0};
            for(auto v_idx : edge_idxs){
                transform += vertexes[v_idx];
            }

            transform = norm(transform);
            float_3 normal = transform;
            transform *= lamp_radius / sin((M_PI - angle_between_edges) / 2.0);

            // add triangles
            triangles.push_back(triangle_to_earth_coords({
                vertexes[edge_idxs[0]] + transform, 
                vertexes[edge_idxs[1]] + transform, 
                vertexes[edge_idxs[2]] + transform, 
                normal,
                material_id
            }));
            triangles.push_back(triangle_to_earth_coords({
                vertexes[edge_idxs[2]] + transform, 
                vertexes[edge_idxs[3]] + transform, 
                vertexes[edge_idxs[4]] + transform, 
                normal,
                material_id
            }));
            triangles.push_back(triangle_to_earth_coords({
                vertexes[edge_idxs[4]] + transform, 
                vertexes[edge_idxs[2]] + transform, 
                vertexes[edge_idxs[0]] + transform, 
                normal,
                material_id
            }));
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

    // init lines and edges description
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


// TODO: Define here another figures

#endif // __FIGURES_CUH__