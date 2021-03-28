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

#define DIST_EPS 5e-4

/*
    Note: Abstract class of 3D Figure

    This class can:
        1. Add triangles and lights to scene with method `render_figures`
        2. It's possible to set radius of figure and position of center
        3. Method `render_figures` generates lights and edges separately
*/
class Figure3d{
protected:
    using point = float_3;
    using line = pair<uint8_t, uint8_t>;
    using edge = vector<uint8_t>;
    using color = float_3;

public:
    /*
        Note: Constructor for Figure

        Takes radius of figure, position of center and material of edges
    */
    Figure3d(float radius, const point& center_position, const material& glass_mat, uint8_t diods_on_line) 
    : figure_radius(radius), coordinats_of_center(center_position), trig_material(glass_mat), diods_per_line(diods_on_line){
        // We will make radius of diod depndent from radius of figure:
        diod_radius *= figure_radius;
    }

    /*
        Note: Generation triangles and lights.

        This class should be called from scene class. Adds triangles and light source to scene.        
    */
    void render_figure(vector<triangle>& triangles, vector<light_point>& light_sources) const{
        split_line_triangles(triangles);
        split_edge_triangles(triangles);
        turn_on_lights(light_sources);
    }

protected:

    /*
        Note: Preprocessing of data

        This method makes computation to generate vertexes and scales them with radius.
    */
    void generate_figure(){
        generate_figure_vertexes();
        scale_vertexes();
    }

    /*
        Note: Scale

        This method scale each vertex with figure radius in local coords.
    */
    void scale_vertexes(){
        for(auto& v : vertexes){
            v *= figure_radius;
        }
    }

    /*
        Note: Moving

        This class moves each triangle to coordinates of figure center.
    */
    triangle triangle_to_earth_coords(const triangle& single_triangle) const{
        triangle result = single_triangle;
        
        result.a += coordinats_of_center; 
        result.b += coordinats_of_center; 
        result.c += coordinats_of_center;
        
        return result;
    }

protected:
    /*
        Note: Vertex generation

        This class generates vertexes of figure with radius equal to 1. 
        For each figure type algorihm diffrent.
    */
    virtual void generate_figure_vertexes() = 0;

    /*
        Note: Edge generation.
        This class generates edges from vertexes and push back to scene vector.
    */
    virtual void split_edge_triangles(vector<triangle>& triangles) const = 0;

protected:
    /*
        Note: Turn lights on.

        In our algorithm diods from lines - just white triangles with real light source in center of figure.
        This method add's this source to scene.
    */
    void turn_on_lights(vector<light_point>& light_sources) const{
        light_sources.push_back({coordinats_of_center, diod_energy_color, diod_energy_power});
    }

    /*
        Note: Lines generation.

        This methods produce lines for figure in 2 steps:
            1. Generation of box for black line.
            2. Adding diods to this box.
    */
    void split_line_triangles(vector<triangle>& triangles) const{
        // Get id of material properties from table.
        uint32_t box_material_id = MaterialTable().get_material_id(material_od_boxes);
        uint32_t diod_material_id = MaterialTable().get_material_id(material_od_diods);


        /*
            Note: Geration of box with diods.

            Decided to make box on small distance from geometrical lines.
            Let's imagine projection of geomterical edge intersection:

                                        /\
                                       /  \
                                      /    \

            We will transform this intersection to following (with diods `o`):
                              y |        _
                               /|\      /o\
                                |      /   \
                                |     /     \       x
                                |__________________\__    
                                                   /

            Therefore it's needed to define lenght of transforming by axes `x` ans `y`:
                1. Transfroming by axes `x` we will define using predefined 
                   constant formula dependent to diod_radius:
                        `x_transform = box_width_coefficent * diod_radius`
                2. If we know `angle` between edges it's possible to find transform by `y`:
                            `y_transform =  box_width * ctg((Pi - angle)/2)`
                    
                    P.S. It's hard to improve this formula here. Just bellive! =)
        */
        for(auto& line_points : lines){
            /*
                Note: Vectors of `x` and `y` axes on projection in local coords:

                Let's imagine line:
                                    line
                             v1 o---------o v2
                                 \_    _ /
                                 |\     /\
                                b  \   /  a
                                    \ /
                                     O(0, 0) - center of local coords

                Therefore `y` axes(vector to mid of lines) can be defined as:
                            `y_transform = a + b`

                                     | y
                                    /|\
                                     |
                             v1 o----|----o v2
                                 \_  | _ /
                                 |\  |  /\
                                b  \ | /  a
                                    \|/
                                     O(0, 0, 0) - center of local coords     z
                                     ___________________________________\___
                                                                        /
                Therefore `x` axes can be defined as:
                    `x_transform = cross(y_transform, v2 - v1) = cross(y_transform, z_transform)` 
            */
            float_3 transform_by_y = norm(vertexes[line_points.first] + vertexes[line_points.second]);
            float_3 transform_by_z = norm(vertexes[line_points.second] - vertexes[line_points.first]);
            float_3 transform_by_x = norm(cross(transform_by_y, transform_by_z));
            {
                float len_of_x_transform = diod_radius * box_width_factor;
                float len_of_y_transfrom = len_of_x_transform / tan((M_PI - angle_between_edges) / 2.0);

                // scale vectors
                transform_by_x *= len_of_x_transform;
                transform_by_y *= len_of_y_transfrom;
            }

            
            /*
                Note: Generate line box.

                Each line - rectangle. Therefor we use 2 triangles.
            */
            triangles.push_back(triangle_to_earth_coords(triangle{
                vertexes[line_points.first] + transform_by_y + transform_by_x, 
                vertexes[line_points.first] + transform_by_y - transform_by_x, 
                vertexes[line_points.second] + transform_by_y - transform_by_x, 
                box_material_id
            }));

            triangles.push_back(triangle_to_earth_coords(triangle{
                vertexes[line_points.second] + transform_by_y + transform_by_x, 
                vertexes[line_points.second] + transform_by_y - transform_by_x, 
                vertexes[line_points.first] + transform_by_y + transform_by_x, 
                box_material_id
            }));

            /*
                Note: Positions of diods.

                We are map diods on line with constant step from start to end position of line.

                Therefore we need to define center position of each diod and create corresponding
                triangle with `diod_radius`:

                                | line box |
                                |          |
                                |    /\    |
                                |   /  \   |
                                |  /    \  |
                                | /_diod_\ |
                                |          |
                                |          |
                                |    /\    |
                                |   /  \   |
                                |  /    \  |
                                | /_diod_\ |
                                |          |

                In order to place diod inside a Figure we reduce vector `transform_y` a little bit.
            */ 
            transform_by_y *= (1 - DIST_EPS);
            transform_by_x = norm(transform_by_x) * diod_radius;
            transform_by_z = norm(transform_by_z) * diod_radius;

            point start_position = vertexes[line_points.first];
            point end_position = vertexes[line_points.second];

            for(uint8_t i = 1; i <= diods_per_line; ++i){
                float_3 add_to_start = (end_position - start_position) * ((float) i) / ((float) (diods_per_line + 1));

                point center_of_diod = start_position + add_to_start + transform_by_y;

                triangles.push_back(triangle_to_earth_coords(triangle{
                    center_of_diod + transform_by_z, 
                    center_of_diod - transform_by_x, 
                    center_of_diod + transform_by_x, 
                    diod_material_id
                }));
            }
        }
    }

protected:
// parameters, defined by user:
    uint8_t diods_per_line;
    const float figure_radius;
    material trig_material;

// constant parameters, defined by developer
    float diod_radius = 0.04f;
    const float box_width_factor = 2.0f;

    const material material_od_boxes = material{color{0, 0, 0}, 0.0f, 0, 0};
    const material material_od_diods = material{color{1.0, 1.0, 1.0}, 1.0, 1.0, 0};

    const color diod_energy_color = color{1.0f, 1.0f, 1.0f};
    const float diod_energy_power = 0.7f;

// constant paramters dependent to Figure type
    vector<line> lines;
    vector<edge> edges;
    vector<point> vertexes;

    float angle_between_edges;
    const point coordinats_of_center;
};

/*
    Note: Class of Tetraeder
*/
class Tetraeder : public Figure3d{
public:
    Tetraeder(float radius_of_figure, const float_3& center_position, const material& glass_material, uint8_t diods_on_line) 
    : Figure3d(radius_of_figure, center_position, glass_material, diods_on_line){
        // Generate Tetraeder data:
        angle_between_edges = acos(1.0 / 3.0);
        set_lines_and_edges_of_figure();
        generate_figure();
    }

private:
    /*
        Note: Vertexes of Tetraeder
    */
    void generate_figure_vertexes() final{
        vertexes.resize(4);

        vertexes[0] = {-2.0f/sqrt(6.0f), -sqrt(2.0f)/3.0f, -1.0/3.0f}; // A
        vertexes[1] = {2.0f/sqrt(6.0f), -sqrt(2.0f)/3.0f, -1.0/3.0f}; // B
        vertexes[2] = {0.0f, 2*sqrt(2.0f)/3.0f, -1.0/3.0f}; // C
        vertexes[3] = {0.0f, 0.0f, 1.0f}; // D
    }

    /*
        Note: Edges of Tetraeder

        This method generates and pushes triangles of edges to scene.
    */
    void split_edge_triangles(vector<triangle>& triangles) const final{
        uint32_t material_id = MaterialTable().get_material_id(trig_material);

        // For each edge:
        for(auto& edge_idxs : edges){
            /*
                Note: Moving to line boxes.

                As described above, we use transform of lines to produce boxes and diods. 
                Here we should transform edges in order to join them with line boxes.

                Let's use parrallel move of all vertexes of edges for that. Direction of transformation move - 
                the center of gravity of edge's vertexes (and in same time ortogonal vector to this edge).

                Distance of transform defined as:
                        `len_of_transform = transform_x / sin((Pi - angle)/2)`

                Value of `transform_x` defined above! Improvement of this formula also deprecated. Please believe =)
            */
    
            // Center of gravity computation:
            float_3 transform = {0.0, 0.0, 0.0};
            for(auto v_idx : edge_idxs){
                transform += vertexes[v_idx];
            }

            {
                float len_of_x_transform = diod_radius * box_width_factor;
                float len_of_transform = len_of_x_transform / sin((M_PI - angle_between_edges) / 2.0);

                transform = norm(transform) * len_of_transform;
            }

            // Info: Uncomment this if you want a gap's beetwen edges and boxes.
            // transform *=  0.0f;

            // Generate triangles:
            triangles.push_back(triangle_to_earth_coords(triangle{
                vertexes[edge_idxs[0]] + transform, 
                vertexes[edge_idxs[1]] + transform, 
                vertexes[edge_idxs[2]] + transform,
                material_id
            }));
        }
    }

private:
    /*
        Note: Lines and Edges

        In generated array of vertexes we can distill folowing lines and edges: 
    */
    void set_lines_and_edges_of_figure(){
        lines = {
            {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}
        };

        edges = {
            {0, 1, 2}, {0, 3, 2},
            {0, 3, 1}, {1, 3, 2}
        };
    }
};


/*
    Note: Class of Octaeder
*/
class Octaeder : public Figure3d{
public:
    Octaeder(float radius_of_figure, const float_3& center_position, const material& glass_material, uint8_t diods_on_line) 
    : Figure3d(radius_of_figure, center_position, glass_material, diods_on_line){
        // Generate Octaeder data:
        angle_between_edges = acos(-1.0 / 3.0);
        set_lines_and_edges_of_figure();
        generate_figure();
    }

private:
    /*
        Note: Vertexes of Octaeder
    */
    void generate_figure_vertexes() final{
        vertexes.resize(6);

        vertexes[0] = {-sqrt(2.0f)/2.0f, -sqrt(2.0f)/2.0f, 0.0f}; // A
        vertexes[1] = {sqrt(2.0f)/2.0f, -sqrt(2.0f)/2.0f, 0.0f}; // B
        vertexes[2] = {sqrt(2.0f)/2.0f, sqrt(2.0f)/2.0f, 0.0f}; // C
        vertexes[3] = {-sqrt(2.0f)/2.0f, sqrt(2.0f)/2.0f, 0.0f}; // D
        vertexes[4] = {0.0f, 0.0f, -1.0f}; // E
        vertexes[5] = {0.0f, 0.0f, 1.0f}; // F
    }

    /*
        Note: Edges of Octaeder

        This method generates and pushes triangles of edges to scene.
    */
    void split_edge_triangles(vector<triangle>& triangles) const final{
        uint32_t material_id = MaterialTable().get_material_id(trig_material);

        // For each edge:
        for(auto& edge_idxs : edges){
            /*
                Note: Moving to line boxes.

                As described above, we use transform of lines to produce boxes and diods. 
                Here we should transform edges in order to join them with line boxes.

                Let's use parrallel move of all vertexes of edges for that. Direction of transformation move - 
                the center of gravity of edge's vertexes (and in same time ortogonal vector to this edge).

                Distance of transform defined as:
                        `len_of_transform = transform_x / sin((Pi - angle)/2)`

                Value of `transform_x` defined above! Improvement of this formula also deprecated. Please believe =)
            */
    
            // Center of gravity computation:
            float_3 transform = {0.0, 0.0, 0.0};
            for(auto v_idx : edge_idxs){
                transform += vertexes[v_idx];
            }

            {
                float len_of_x_transform = diod_radius * box_width_factor;
                float len_of_transform = len_of_x_transform / sin((M_PI - angle_between_edges) / 2.0);

                transform = norm(transform) * len_of_transform;
            }

            // Info: Incomment this if you want a gap's beetwen edges and boxes.
            // transform *=  0.0f;


            // Generate triangles:
            triangles.push_back(triangle_to_earth_coords(triangle{
                vertexes[edge_idxs[0]] + transform, 
                vertexes[edge_idxs[1]] + transform, 
                vertexes[edge_idxs[2]] + transform,
                material_id
            }));
        }
    }

private:
    /*
        Note: Lines and Edges

        In generated array of vertexes we can distill folowing lines and edges: 
    */
    void set_lines_and_edges_of_figure(){
        lines = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, 
            {0, 4}, {1, 4}, {2, 4}, {3, 4},
            {0, 5}, {1, 5}, {2, 5}, {3, 5}
        };

        edges = {
            {0, 4, 1}, {1, 4, 2}, {2, 4, 3}, {3, 4, 0},
            {0, 5, 1}, {1, 5, 2}, {2, 5, 3}, {3, 5, 0}
        };
    }
};


/*
    Note: Class of Dodecaedr
*/
class Dodecaedr : public Figure3d{
public:
    Dodecaedr(float radius_of_figure, const float_3& center_position, const material& glass_material, uint8_t diods_on_line) 
    : Figure3d(radius_of_figure, center_position, glass_material, diods_on_line){
        // Generate Dodecaedr data:
        angle_between_edges = 2.034443043;
        set_lines_and_edges_of_figure();
        generate_figure();
    }

private:
    /*
        Note: Vertexes of Dodecaedr

        Generation of Dodecaedr's vertexes based on icosaedr vertexes.
        ref: https://works.doklad.ru/view/1SxrxyeGZoA.html
    */
    void generate_figure_vertexes() final{
        vertexes.resize(20);

        const vector<uint8_t> G0 = {0, 2, 4, 6, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 3, 5, 7, 9};
        const vector<uint8_t> G1 = {10, 10, 10, 10, 10, 2, 2, 4, 4, 6, 6, 8, 8, 0, 0, 11, 11, 11, 11, 11};
        const vector<uint8_t> G2 = {2, 4, 6, 8, 0, 1, 3, 3, 5, 5, 7, 7, 9, 9, 1, 9, 1, 3, 5, 7};

        vector<point> ico_vertexes;
        generate_icosaedr_vertexes(ico_vertexes);

        // coords of Dodecaedr:
        for(uint8_t i = 0; i < 20; ++i){
            vertexes[i] = ico_vertexes[G0[i]] + ico_vertexes[G1[i]] + ico_vertexes[G2[i]];
            vertexes[i] /= 3.0f;
        }
    }

    /*
        Note: Edges of Dodecaedr

        This method generates and pushes triangles of edges to scene.
    */
    void split_edge_triangles(vector<triangle>& triangles) const final{
        uint32_t material_id = MaterialTable().get_material_id(trig_material);

        // For each edge:
        for(auto& edge_idxs : edges){
            /*
                Note: Moving to line boxes.

                As described above, we use transform of lines to produce boxes and diods. 
                Here we should transform edges in order to join them with line boxes.

                Let's use parrallel move of all vertexes of edges for that. Direction of transformation move - 
                the center of gravity of edge's vertexes (and in same time ortogonal vector to this edge).

                Distance of transform defined as:
                        `len_of_transform = transform_x / sin((Pi - angle)/2)`

                Value of `transform_x` defined above! Improvement of this formula also deprecated. Please believe =)
            */
    
            // Center of gravity computation:
            float_3 transform = {0.0, 0.0, 0.0};
            for(auto v_idx : edge_idxs){
                transform += vertexes[v_idx];
            }

            {
                float len_of_x_transform = diod_radius * box_width_factor;
                float len_of_transform = len_of_x_transform / sin((M_PI - angle_between_edges) / 2.0);

                transform = norm(transform) * len_of_transform;
            }

            // Info: Uncomment this if you want a gap's beetwen edges and boxes.
            // transform *=  0.0f;


            // Generate triangles:
            triangles.push_back(triangle_to_earth_coords(triangle{
                vertexes[edge_idxs[0]] + transform, 
                vertexes[edge_idxs[1]] + transform, 
                vertexes[edge_idxs[2]] + transform,
                material_id
            }));
            triangles.push_back(triangle_to_earth_coords(triangle{
                vertexes[edge_idxs[2]] + transform, 
                vertexes[edge_idxs[3]] + transform, 
                vertexes[edge_idxs[4]] + transform, 
                material_id
            }));
            triangles.push_back(triangle_to_earth_coords(triangle{
                vertexes[edge_idxs[4]] + transform, 
                vertexes[edge_idxs[2]] + transform, 
                vertexes[edge_idxs[0]] + transform, 
                material_id
            }));
        }
    }

private:
    /* 
        Note: Icosaedr

        This function generates vertexes of icosaedr with radius equal to 1.
    */
    void generate_icosaedr_vertexes(vector<point>& ico_vertxs) const{
        ico_vertxs.resize(12);

        float h = 0.5;
        const float Ro = 1.118034;
        
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

    /*
        Note: Lines and Edges

        In generated array of vertexes we can distill folowing lines and edges: 
    */
    void set_lines_and_edges_of_figure(){
        lines = {
            {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 5}, {5, 6},
            {6, 7}, {7, 1}, {8, 7}, {9, 8}, {2, 9}, {9, 10}, {10, 11},
            {11, 3}, {12, 11}, {13, 12}, {4, 13}, {13, 14}, {14, 5}, {15, 14},
            {16, 15}, {6, 16}, {16, 17}, {17, 8}, {18, 17}, {10, 18}, {18, 19},
            {19, 12}, {15, 19}
        };

        edges = {
            {0, 1, 2, 3, 4}, {15, 16, 17, 18, 19},
            {6, 7, 8, 17, 16}, {6, 5, 14, 15, 16},
            {12, 13, 14, 15, 19}, {18, 19, 12, 11, 10},
            {10, 9, 8, 17, 18}, {0, 1, 7, 6, 5},
            {4, 0, 5, 14, 13}, {13, 4, 3, 11, 12},
            {11, 3, 2, 9, 10}, {8, 9, 2, 1, 7}
        };
    }
};


/*
    Note: Class of Simple Cube

    It's made for testing. May be removed!
*/
class Cube : public Figure3d{
public:
    Cube(float radius_o, const float_3& position, const material& glass_mat, uint8_t line_lamps) 
    : Figure3d(radius_o, position, glass_mat, line_lamps){
        angle_between_edges = M_PI_2;
                        
        // generate Dodecaedr
        generate_figure();
    }

private:
    // this function generates vertexes of Dodecaedr with R = 1
    void generate_figure_vertexes() final{
        vertexes = {
            {-1, -1, -1}, // 0
            {-1, -1, 1}, // 1
            {-1, 1, -1}, // 2
            {-1, 1, 1}, // 3
            {1, -1, -1}, // 4
            {1, -1, 1}, // 5
            {1, 1, -1}, // 6
            {1, 1, 1}  // 7
        };
    }

    void split_edge_triangles(vector<triangle>& triangles) const final{
        uint32_t material_id = MaterialTable().get_material_id(trig_material);

        // 1
        triangles.push_back(triangle_to_earth_coords({
            vertexes[0], 
            vertexes[1], 
            vertexes[3], 
            material_id
        }));
        triangles.push_back(triangle_to_earth_coords({
            vertexes[0], 
            vertexes[2], 
            vertexes[3], 
            material_id
        }));

        // 2
        triangles.push_back(triangle_to_earth_coords({
            vertexes[1], 
            vertexes[5], 
            vertexes[7], 
            material_id
        }));
        triangles.push_back(triangle_to_earth_coords({
            vertexes[1], 
            vertexes[3], 
            vertexes[7], 
            material_id
        }));

        // 3
        triangles.push_back(triangle_to_earth_coords({
            vertexes[4], 
            vertexes[5], 
            vertexes[7], 
            material_id
        }));
        triangles.push_back(triangle_to_earth_coords({
            vertexes[4], 
            vertexes[6], 
            vertexes[7], 
            material_id
        }));

        // 4
        triangles.push_back(triangle_to_earth_coords({
            vertexes[0], 
            vertexes[4], 
            vertexes[6], 
            material_id
        }));
        triangles.push_back(triangle_to_earth_coords({
            vertexes[0], 
            vertexes[2], 
            vertexes[6], 
            material_id
        }));

        // 5
        triangles.push_back(triangle_to_earth_coords({
            vertexes[0], 
            vertexes[1], 
            vertexes[5], 
            material_id
        }));
        triangles.push_back(triangle_to_earth_coords({
            vertexes[0], 
            vertexes[4], 
            vertexes[5], 
            material_id
        }));

        // 6
        triangles.push_back(triangle_to_earth_coords({
            vertexes[2], 
            vertexes[3], 
            vertexes[7], 
            material_id
        }));
        triangles.push_back(triangle_to_earth_coords({
            vertexes[2], 
            vertexes[6], 
            vertexes[7], 
            material_id
        }));
    }

private:
    // init lines and edges description
    void set_lines_and_edges_of_figure(){
        // pass
    }
};



// TODO: Define here another figures

#endif // __FIGURES_CUH__