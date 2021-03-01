// Made by Max Bronnikov
#ifndef __MATERIAL_TABLE_CUH__
#define __MATERIAL_TABLE_CUH__

#include <unordered_map>
#include <vector>
#include "structures.cuh"

// simple singleton class for definition of id for each material
class MaterialTable{
public:
    MaterialTable(){}

    uint32_t get_material_id(const material& mat){
        if(table.count(mat)){
            return table[mat];
        }

        uint32_t size = table.size();
        return table[mat] = size;
    }

    void save_to_vector(std::vector<material> vec){
        vec.resize(table.size());
        for(auto it : table){
            vec[it.second] = it.first;
        }
    }

private:
    static std::unordered_map<material, uint32_t, material_functor> table;
};

std::unordered_map<material, uint32_t, material_functor> MaterialTable::table;



#endif //__MATERIAL_TABLE_CUH__
