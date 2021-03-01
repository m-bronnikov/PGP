#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

struct float_3
{   
    float x;
    float y;
    float z;
};


struct material{
    float_3 color;
    float diffussion;
    float reflection;
    float refraction;
};


bool operator==(const material& lhs, const material& rhs){
    return lhs.color.x == rhs.color.x && lhs.color.y == rhs.color.y && lhs.color.z == rhs.color.z &&
    lhs.diffussion == rhs.diffussion && lhs.reflection == rhs.reflection && lhs.refraction == rhs.refraction;
}


struct hash_functor
{
    size_t operator()(const material& mat) const{
        return static_cast<size_t>(
            103245.0 * (mat.color.x + mat.color.y + mat.color.z + mat.diffussion + mat.refraction + mat.reflection)
        );
    }
};


class MaterialTable{
public:
    MaterialTable(){}

    uint32_t get_material_id(const material& mat){
        uint32_t size = table.size();

        if(table.count(mat)){
            return table[mat];
        }

        return table[mat] = size;
    }

    void save_to_vector(std::vector<material> vec){
        vec.resize(table.size());
        for(auto it : table){
            vec[it.second] = it.first;
        }
    }

private:
    static std::unordered_map<material, uint32_t, hash_functor> table;
};

std::unordered_map<material, uint32_t, hash_functor> MaterialTable::table;



using namespace std;

int main(){
    uint32_t a = 5;

    a = MaterialTable().get_material_id({{0.85, 0.85, 0.85}, 0.8, 0.9, 0});

    cout << a << endl;

    return 0;
}