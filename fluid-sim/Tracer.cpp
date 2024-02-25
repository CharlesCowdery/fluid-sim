// spatialPartitioning.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define _AMD64_ true
#define _WIN64 true
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#include <tuple>
#include <math.h>
#include <format>
#include <string>
#include <ctime>
#include <chrono>
#include <functional>
#include <fstream>

#include <SFML/Graphics.hpp>
#include <OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;

#include <thread>
#include <vector>
#include <array>
#include <unordered_set>
#include <set>
#include <queue>
#include <processthreadsapi.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cassert>
#include <immintrin.h>

#include <boost/array.hpp>
#include <boost/asio.hpp>

import XYZ;
import XorRandom;
import VecLib;
import Matrix;
import Materials;

#define assertm(exp, msg) assert(((void)msg, exp))
#define SMALL 0.001
#define NEAR_THRESHOLD 0.000001
#define SCENE_BOUNDS 10000
#define RED_MASK 255<<16
#define GREEN_MASK 255<<8
#define BLUE_MASK 255

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

#define PI 3.14159265358979323846264

#define kEpsilon 0.000000001

#define DRAW_COLOR false
#define DEPTH_VIEW false
#define DRAW_UV false
#define DRAW_BOUNCE_DIRECTION false
#define DRAW_NORMAL false
#define DRAW_EMISSIVE false
#define GENERATED_TEXTURE_RESOLUTION 2048

#define USE_ADVANCED_BVH false

//defines program precision

#define decimal float

#define USE_AVX_BVH 1
#define USE_AVX_TRI 1
#define ENABLE_MIS  0
#define USE_PACKET_TRAVERSAL 0

#if USE_AVX_TRI 1
#define LEAF_SIZE 8
#define PENALIZE_UNFILLED_LEAFS 1
#else
#define LEAF_SIZE 2
#define PENALIZE_UNFILLED_LEAFS 0
#endif


using namespace std;
using boost::asio::ip::tcp;
using time_point = chrono::steady_clock::time_point;


static thread_local const __m256 AVX_ZEROS = _mm256_set1_ps(0);

static thread_local XorGen gen;


/*Xorshiro256+ pseudorandom end*/


decimal clamp(decimal value, decimal low, decimal high) {
    return max(min(value, high), low);
}

decimal lerp(decimal a, decimal b, decimal factor) {
    return a + factor * (b - a);
}

vector<string> eng_levels = vector<string>{
    "k",
    "M",
    "B",
    "T",
    "qu",
    "Qu"
};

string padString(string s, string padder, int size) {
    string out = s;
    while (out.size() < size) {
        out += padder;
    }
    return out;
}

string iToFixedLength(int input, int length, string fill = " ") {
    string in = to_string(input);
    int spaces = length - in.size();
    for (int i = 0; i < spaces;i++) {
        in = fill + in;
    }
    return in;
}
string intToEng(int inum, bool force_small = false) {
    double num = inum;
    string prefix = "";
    if (num < 1000) {
        return iToFixedLength(inum, 4) + " ";
    }
    for (auto level : eng_levels) {
        if (num < 1000) {
            string number = to_string(num);
            if (num >= 100) {
                number = number.substr(0, 3);
                if (!force_small) {
                    number = " " + number;
                }
            }
            else {
                if (!force_small) {
                    number = number.substr(0, 4);
                }
                else {
                    number = number.substr(0, 3);
                }
            }
            return number + prefix;
        }
        num /= 1000;
        prefix = level;
    }
}

vector<string> split_string(string input, char splitter) {
    vector<string> output;
    string s;
    int size = input.size();
    for (int i = 0; i < size; i++) {
        char c = input[i];
        if (c == splitter) {
            output.push_back(s);
            s = "";
        }
        else {
            s += c;
        }
    }
    output.push_back(s);
    return output;
}



typedef tuple<XYZ, XYZ, XYZ> Face;
typedef tuple<bool, bool, bool> tri_bool;


class Primitive {
public:
    //pair<XYZ, XYZ> bounds;
    XYZ origin;
    int obj_type = 0;
    Primitive() {}
};

class PointLikeLight {
public:
    XYZ origin;
    decimal radius;
    Primitive* host;
    PointLikeLight(XYZ _origin, decimal _radius, Primitive* _host) : origin(_origin), radius(_radius), host(_host) {}
};

class SpotLight {
    XYZ origin;
    XYZ pointing;
    Primitive* host;
    SpotLight(XYZ _origin, XYZ _pointing, Primitive* _host) :origin(_origin), pointing(_pointing), host(_host) {}
    static SpotLight fromNormal(XYZ origin, XYZ normal, Primitive* host) {
        return SpotLight(origin, XYZ::flip(normal), host);
    }
};



struct PackagedTri { //reduced memory footprint to improve cache perf
    XYZ p1;
    XYZ normal;
    XYZ p1p3;
    XYZ p1p2;
    XYZ origin_offset;
    PackagedTri() {}
    PackagedTri(const XYZ& _p1, const XYZ& _p2, const XYZ& _p3) {
        p1 = _p1;
        p1p3 = _p3 - p1;
        p1p2 = _p2 - p1;
        normal = XYZ::normalize(XYZ::cross(p1p3, p1p2));
        origin_offset = XYZ::dot((_p1 + _p2 + _p3) / 3, normal) * normal;
    }
};

struct PTri_AVX { //literally 0.7kB of memory per object. What have I done.
    m256_vec3 p1;
    m256_vec3 p1p2;
    m256_vec3 p1p3;
    m256_vec3 normal;
    m256_vec3 origin_offset;
    char size = 0;
    PTri_AVX(vector<PackagedTri> tris) {
        vector<XYZ> _p1;
        vector<XYZ> _p1p2;
        vector<XYZ> _p1p3;
        vector<XYZ> _normal;
        vector<XYZ> _origin_offset;
        size = tris.size();
        for (int i = 0; i < tris.size() && i < 8; i++) {
            PackagedTri& tri = tris[i];
            _p1.push_back(tri.p1);
            _p1p2.push_back(tri.p1p2);
            _p1p3.push_back(tri.p1p3);
            _normal.push_back(tri.normal);
            _origin_offset.push_back(tri.origin_offset);
        }
        p1 = m256_vec3(_p1);
        p1p2 = m256_vec3(_p1p2);
        p1p3 = m256_vec3(_p1p3);
        normal = m256_vec3(_normal);
        origin_offset = m256_vec3(_origin_offset);
    }
    static void intersection_check(const PTri_AVX& T, const m256_vec3& position, const m256_vec3& slope, m256_vec3& output) { //my FUCKING cache 
        m256_vec3 pvec;
        VecLib::cross_avx(slope, T.p1p3, pvec);
        __m256 det;
        VecLib::dot_avx(T.p1p2, pvec, det);

        __m256 invDet = _mm256_div_ps(_mm256_set1_ps(1), det);

        m256_vec3 tvec;
        m256_vec3::sub(position, T.p1, tvec);
        m256_vec3 qvec;
        VecLib::cross_avx(tvec, T.p1p2, qvec);

        m256_vec2 uv;
        VecLib::dot_mul_avx(tvec, pvec, invDet, uv.X);
        VecLib::dot_mul_avx(slope, qvec, invDet, uv.Y);
        VecLib::dot_mul_avx(T.p1p3, qvec, invDet, output.X); //t

        __m256 u_pass = _mm256_cmp_ps(uv.X, AVX_ZEROS, _CMP_GE_OQ);
        __m256 v_pass = _mm256_cmp_ps(uv.Y, AVX_ZEROS, _CMP_GE_OQ);
        __m256 uv_pass = _mm256_cmp_ps(
            _mm256_add_ps(
                uv.X,
                uv.Y
            ),
            _mm256_set1_ps(1),
            _CMP_LE_OQ
        );

        output.X = _mm256_and_ps(
            uv_pass,
            output.X
        );

        output.X = _mm256_and_ps(
            u_pass,
            output.X
        );

        output.X = _mm256_and_ps(
            v_pass,
            output.X
        );

        output.Y = uv.X;
        output.Z = uv.Y;
    }
    static void get_normal(const PTri_AVX& T, const m256_vec3& position, m256_vec3& output) {
        __m256 dot;
        m256_vec3 relative_position;
        m256_vec3::sub(position, T.origin_offset, relative_position);
        VecLib::dot_avx(T.normal, relative_position, dot); //this can probably be done faster via xor of sign bit rather than multiplication, but whatever
        __m256 sgn = VecLib::sgn_fast(dot);
        output.X = _mm256_mul_ps(
            T.normal.X,
            sgn
        );
        output.Y = _mm256_mul_ps(
            T.normal.Y,
            sgn
        );
        output.Z = _mm256_mul_ps(
            T.normal.Z,
            sgn
        );
    }
};

 class Tri :public Primitive {
public:
    XYZ p1;
    XYZ p2;
    XYZ p3;
    XYZ midpoint;
    XYZ AABB_max;
    XYZ AABB_min;
    Tri(XYZ _p1, XYZ _p2, XYZ _p3) : Primitive(),
        p1(_p1), p2(_p2), p3(_p3) {
        midpoint = (p1 + p2 + p3) / 3;
        AABB_max = XYZ::max(p1, XYZ::max(p2, p3));
        AABB_min = XYZ::min(p1, XYZ::min(p2, p3));
    }
    //https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
    static XYZ intersection_check_MoTru(const PackagedTri& T, const XYZ& position, const XYZ& slope) {
        XYZ pvec = XYZ::cross(slope, T.p1p3);
        decimal det = XYZ::dot(T.p1p2, pvec);
#if BACKFACE_CULLING
        // if the determinant is negative, the triangle is 'back facing'
        // if the determinant is close to 0, the ray misses the triangle
        (det < kEpsilon) return XYZ(-1);
#else
        // ray and triangle are parallel if det is close to 0
        if (fabs(det) < kEpsilon) return -1;
#endif
        decimal invDet = 1 / det;

        XYZ tvec = position - T.p1;
        decimal u = XYZ::dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1) return XYZ(-1);

        XYZ qvec = XYZ::cross(tvec, T.p1p2);
        decimal v = XYZ::dot(slope, qvec) * invDet;
        if (v < 0 || u + v > 1) return XYZ(-1);

        decimal t = XYZ::dot(T.p1p3, qvec) * invDet;
        return XYZ(t, u,v);
    }
    static XYZ intersection_check(const PackagedTri& T, const XYZ& position, const XYZ& slope) {
        return intersection_check_MoTru(T, position, slope);
    }
    static XYZ random(const PackagedTri& T) {
        float r1 = gen.fRand(0, 1);
        float r2 = gen.fRand(0, 1);
        if (r1 + r2 > 1) {
            r1 = 1 - r1;
            r2 = 1 - r2;
        }
        return T.p1 + r1 * T.p1p2 + r2 * T.p1p3;
    }
    static XYZ get_normal(XYZ normal, XYZ relative_position) {
        return (XYZ::dot(normal, relative_position) > 0) ? normal : -normal;
    }
    PackagedTri pack() {
        return PackagedTri(
            p1, p2, p3
        );
    }
};


class Transformation {
private:
    Quat* rot_transform = nullptr;
    XYZ* XYZ_transform = nullptr;
public:
    Transformation(XYZ transformation, Quat rotation) {
        rot_transform = new Quat(rotation);
        XYZ_transform = new XYZ(transformation);
    }
    Transformation(Quat rotation) {
        rot_transform = new Quat(rotation);
    }
    Transformation(XYZ transformation) {
        XYZ_transform = new XYZ(transformation);
    }
    void stack(XYZ& XYZ_transf, Quat& rotation) {
        if (XYZ_transform != nullptr) {
            XYZ_transf += *XYZ_transform;
        }
        if (rot_transform != nullptr) {
            rotation = Quat::multiply(rotation, *rot_transform);
        }
    }
    //takes an XYZ position reference, and applies itself to it
    void apply(XYZ& position) {
        if (rot_transform != nullptr) {
            position = Quat::applyRotation(position, *rot_transform);
        }
        if (XYZ_transform != nullptr) {
            position += *XYZ_transform;
        }
    }
    static Transformation collpase(vector<Transformation>& transforms) {
        XYZ position = XYZ(0, 0, 0);
        Quat rotation = Quat(0, 0, 0, 1);
        for (Transformation& t : transforms) {
            t.stack(position, rotation);
        }
        return Transformation(position, rotation);
    }
};

namespace Packers { //I wanted to put these methods in the primitives objects, but due to me not using header files atm, its not feasible
    PackagedTri transformedPack(Tri& t, Transformation T, XYZ origin, XYZ scale) {
        XYZ np1 = t.p1 - origin;
        XYZ np2 = t.p2 - origin;
        XYZ np3 = t.p3 - origin;
        np1 *= scale;
        np2 *= scale;
        np3 *= scale;
        T.apply(np1);
        T.apply(np2);
        T.apply(np3);
        np1 += origin;
        np2 += origin;
        np3 += origin;
        return PackagedTri(
            np1, np2, np3
        );
    }
    Tri transformT(Tri& t, Transformation T, XYZ origin, XYZ scale) {
        XYZ np1 = t.p1 - origin;
        XYZ np2 = t.p2 - origin;
        XYZ np3 = t.p3 - origin;
        np1 *= scale;
        np2 *= scale;
        np3 *= scale;
        T.apply(np1);
        T.apply(np2);
        T.apply(np3);
        np1 += origin;
        np2 += origin;
        np3 += origin;
        return Tri(
            np1, np2, np3
        );
    }
}

 class Mesh {
public:
    vector<Tri> tris;
    string name = "";
    int primitive_count;
    Mesh() {}
    void prep() {
    }
    void addTri(Tri& T) {
        tris.push_back(T);
        primitive_count++;
    }
};

 class Object {
public:
    XYZ origin;
    XYZ scale;
    vector<Transformation> transformations;

    vector<Object*> children;

    vector<Mesh*> meshes;

    string name = "";

    Object(XYZ _origin, XYZ _scale) :
        origin(_origin), scale(_scale) {

    }
    void addMesh(Mesh* M) {
        meshes.push_back(M);
    }

    void addChild(Object* object) {
        children.push_back(object);
    }
    void prep() {

        for (Mesh* m : meshes) {
            m->prep();
        }
    }
    void fetchData( vector<Tri>& tris_vec) {
        Transformation T = final_transform();
        for (Mesh* M : meshes) {
            for (Tri& tri : M->tris) {
                tris_vec.push_back(Packers::transformT(tri, T, origin, scale));
            }
        }
        for (Object* child : children) {
            child->fetchData(tris_vec);
        }
    }
    void _registerRotation(Quat rot) {
        transformations.push_back(Transformation(rot));
    }
    void _registerMove(XYZ move) {
        transformations.push_back(Transformation(move));
    }
    void applyTransformXYZ(decimal x, decimal y, decimal z) {
        _registerMove(XYZ(x, y, z));
    }
    void applyTransformXYZ(XYZ xyz) {
        applyTransformXYZ(xyz.X, xyz.Y, xyz.Z);
    }
    void rotateX(decimal rotation) {
        XYZ orig = XYZ(0, 1, 0);
        XYZ pointing = XYZ(0, cos(rotation), sin(rotation));
        Quat rot = Quat::makeRotation(orig, pointing);
        _registerRotation(rot);
    }
    void rotateY(decimal rotation) {
        XYZ orig = XYZ(0, 0, 1);
        XYZ pointing = XYZ(sin(rotation), 0, cos(rotation));
        Quat rot = Quat::makeRotation(orig, pointing);
        _registerRotation(rot);
    }
    void rotateZ(decimal rotation) {
        XYZ orig = XYZ(1, 0, 0);
        XYZ pointing = XYZ(cos(rotation), sin(rotation), 0);
        Quat rot = Quat::makeRotation(orig, pointing);
        _registerRotation(rot);
    }
    Transformation final_transform() {
        return Transformation::collpase(transformations);
    }
};





//http://bannalia.blogspot.com/2015/06/cache-friendly-binary-search.html
int tree_total = 0;
int leaf_total = 0;
decimal shortest = 99;

class BVH {
public:
    XYZ max = XYZ(0, 0, 0);
    XYZ min = XYZ(0, 0, 0);
    vector<Tri*> elements;
    BVH* c1 = nullptr;
    BVH* c2 = nullptr;
    BVH* parent;
    int _count = -1;
    BVH() {    }
    BVH(vector<Tri*>* T_vec) {
        elements = vector<Tri*>(*T_vec);
        leaf_total += T_vec->size();
        for (Tri* T_ptr : *T_vec) {
            max = XYZ::max(max, T_ptr->AABB_max);
            min = XYZ::min(min, T_ptr->AABB_min);
        }

    }
    ~BVH() {
        delete c1;
        delete c2;
    }
    static decimal intersection(const XYZ& max, const XYZ& min, const XYZ& origin, const XYZ& inv_slope) {
        decimal tx1 = (min.X - origin.X) * inv_slope.X;
        decimal tx2 = (max.X - origin.X) * inv_slope.X;

        decimal tmin = std::min(tx1, tx2);
        decimal tmax = std::max(tx1, tx2);

        decimal ty1 = (min.Y - origin.Y) * inv_slope.Y;
        decimal ty2 = (max.Y - origin.Y) * inv_slope.Y;

        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));

        decimal tz1 = (min.Z - origin.Z) * inv_slope.Z;
        decimal tz2 = (max.Z - origin.Z) * inv_slope.Z;

        tmin = std::max(tmin, std::min(tz1, tz2));
        tmax = std::min(tmax, std::max(tz1, tz2));

        if (tmax >= tmin) { //introduces a branch, but itll probably be fine. Lets me order search checks by nearest intersection
            if (tmin >= 0) {
                return tmin;
            }
            else {
                return tmax;
            }
        }
        else {
            return -1;
        }
    }
    decimal intersection(const XYZ& origin, const XYZ& inv_slope) { //https://tavianator.com/2011/ray_box.html
        return BVH::intersection(max, min, origin, inv_slope);
    }

    struct WorkPacket {
        BVH* target;
        vector<Tri*>* contents;
        WorkPacket(BVH* t, vector<Tri*>* c) : target(t), contents(c) {}
    };
    //very dangerous. only ever use if BVH is being used as an intermediate
    //will turn every tri into a hanging pointer if the passed vector moves out of scope
    int construct_dangerous(vector<Tri>& initial_geo) {
        vector<Tri*>* working_data = new vector<Tri*>();
        working_data->reserve(initial_geo.size());
        for (Tri& T : initial_geo) {
            working_data->push_back(&T);
        }
        int return_value = construct(working_data);
        return return_value;
    }
    int construct(vector<Tri*>* initial_geo) {
        queue<WorkPacket> packets;
        assertm((initial_geo->size() > 0), "BVH was creation was attempted with zero geometry. Did everything load right?");
        auto initial_packet = WorkPacket(this, initial_geo);
        packets.push(initial_packet);
        int i = 0;
        while (packets.size() > 0) {
            i++;
            WorkPacket packet = packets.front();
            vector<Tri*>* geo = packet.contents;
            BVH* target = packet.target;

            assertm(geo->size() > 0, "zero size bin");

            AABB my_bounds = getBounds(geo);

            target->max = my_bounds.max;
            target->min = my_bounds.min;

            XYZ extent = target->max - target->min;
            int axis = 0;
            if (extent.Y > extent.X) axis = 1;
            if (extent.Z > extent[axis]) axis = 2;
            float splitPos = target->min[axis] + extent[axis] * 0.5f;

            Split* split = new Split(splitPos, axis);
            get_stats(geo, *split);
            evaluate_split(*split, 1);
            Split* probe_split = probe(geo);
            get_stats(geo, *probe_split);
            evaluate_split(*probe_split, 1);


            if (probe_split->score < split->score) {
                delete split;
                split = probe_split;
            }
            auto bins = bin(split, geo);
            //if (geo->size() > 200) {
            //    auto binned_split = binned_split_probe(geo);
            //    if (binned_split.first->score < split->score) {
            //        bins = binned_split.second;
            //        split = binned_split.first;
            //    }
            //}


            if (((size_t)abs((int)(bins.p_geo->size() - bins.n_geo->size()))) == geo->size()) {
                target->elements = vector<Tri*>(*geo);
                leaf_total += geo->size();
                packets.pop();
                continue;
            }

            if (bins.p_geo->size() <= LEAF_SIZE) {
                assertm(bins.p_geo->size() > 0, "attempted bin creation of zero elements");
                target->c1 = new BVH(bins.p_geo);
                target->c1->parent = target;
            }
            else {
                BVH* leaf = new BVH();
                target->c1 = leaf;
                target->c1->parent = target;
                WorkPacket k = WorkPacket(leaf, bins.p_geo);
                packets.push(k);
            }
            if (bins.n_geo->size() <= LEAF_SIZE) {
                assertm(bins.n_geo->size() > 0, "attempted bin creation of zero elements");
                target->c2 = new BVH(bins.n_geo);
                target->c2->parent = target;
            }
            else {
                BVH* leaf = new BVH();
                target->c2 = leaf;
                target->c2->parent = target;
                WorkPacket k = WorkPacket(leaf, bins.n_geo);
                packets.push(k);
            }

            delete geo;
            packets.pop();
        }
        return i;
    }
    pair<vector<BVH*>*, vector<Tri*>*> flatten(int depth = 0) {
        vector<BVH*>* l = new vector<BVH*>();
        vector<Tri*>* t = new vector<Tri*>();

        flatten(l, t, depth);
        return pair<vector<BVH*>*, vector<Tri*>*>(l, t);
    }
    void flatten(vector<BVH*>* l, vector<Tri*>* t, int depth) {
        depth = depth - 1;
        l->push_back(this);
        if (depth != 0) {
            if (c1 != nullptr) {
                c1->flatten(l, t, depth);
            }
            if (c2 != nullptr) {
                c2->flatten(l, t, depth);
            }
            if (elements.size() > 0) {
                for (auto T : elements) {
                    t->push_back(T);
                }
            }
        }
    }
    int size() {
        int s = 0;
        size(s);
        return s;
    }
    void size(int& s) {
        s++;
        if (c1 != nullptr) {
            c1->size(s);
            c2->size(s);
        }
    }
    int count() {
        if (_count == -1) {
            count(_count);
        }
        return _count;
    }
    void count(int& s) const {
        s += this->elements.size();
        if (c1 != nullptr) {
            c1->count(s);
            c2->count(s);
        }
    }
    float avg_path() const {
        if (c1 == nullptr) return 0;
        return (c1->avg_path() + c2->avg_path()) / 2.0;
    }
    vector<PackagedTri> get_emissive_tris() {
        vector<PackagedTri> out;
        get_emissive_tris(out);
        return out;
    }
    void get_emissive_tris(vector<PackagedTri>& vec) {
        if (c1 != nullptr) {
            c1->get_emissive_tris(vec);
            c2->get_emissive_tris(vec);
        }
        else {
            for (int i = 0; i < elements.size(); i++) {
                Tri* t = elements[i];
            }
        }
    }

private:
    struct BinResults {
        vector<Tri*>* p_geo;
        vector<Tri*>* n_geo;
    };
    struct Stats {
        XYZ min = XYZ(99999999999);
        XYZ max = XYZ();
        int count = 0;
        decimal SA() {
            return VecLib::surface_area(max, min);
        }
        decimal volume() {
            return VecLib::volume(max, min);
        }
    };
    struct Split {
        Stats p;
        Stats n;
        decimal score;
        XYZ placement;
        int facing;
        Split(XYZ place, int _facing) :placement(place), facing(_facing) {};
    };
    struct AABB {
        XYZ max;
        XYZ min;
        AABB() : max(-99999999999999), min(99999999999999) {}
        AABB(XYZ _max, XYZ _min) :max(_max), min(_min) {}
        void expand(XYZ point) {
            max = XYZ::max(max, point);
            min = XYZ::min(min, point);
        }
    };
    struct reduced_tri {
        XYZ aabb_max;
        XYZ aabb_min;
        XYZ midpoint;
        reduced_tri(Tri T) {
            aabb_max = T.AABB_max;
            aabb_min = T.AABB_min;
            midpoint = T.midpoint;
        }
        struct less_than_axis_operator {
            int axis;
            less_than_axis_operator(int _axis) {
                axis = _axis;
            }
            inline bool operator() (const reduced_tri t1, const reduced_tri t2)
            {
                return (t1.midpoint[axis] < t2.midpoint[axis]);
            }
        };
    };
    AABB getBounds(vector<Tri*>* geo) {
        AABB results;
        for (Tri* T_ptr : *geo) {
            results.expand(T_ptr->AABB_max);
            results.expand(T_ptr->AABB_min);
        }
        return results;
    }
    BinResults bin(Split* split, vector<Tri*>* geo) {
        BinResults results;
        results.p_geo = new vector<Tri*>();
        results.n_geo = new vector<Tri*>();
        int axis = split->facing;
        for (Tri* T_ptr : *geo) {
            if (T_ptr->midpoint[axis] > split->placement[axis]) {
                results.p_geo->push_back(T_ptr);
            }
            else {
                results.n_geo->push_back(T_ptr);
            }
        }
        return results;
    }
    struct relevant_value {
        XYZ midpoint;
        XYZ max;
        XYZ min;
        XYZ right_max;
        XYZ right_min;
        Tri* parent;
        int value_selector = 0;
        relevant_value(Tri* source, int v_selector = 0) {
            parent = source;
            midpoint = parent->midpoint;
            max = parent->AABB_max;
            min = parent->AABB_min;
            value_selector = v_selector;
        }
        XYZ get_value() const {
            switch (value_selector) {
            case 0:
                return min;
            case 1:
                return midpoint;
            case 2:
                return max;
            default:
                throw exception("out of bounds value selector");
            }
        }
        struct comparer {
        public:
            comparer(char compare_index, bool _use_selector = false) {
                internal_index = compare_index;
                use_selector = _use_selector;
            }
            inline bool operator()(const relevant_value& v1, const relevant_value& v2) {
                if (use_selector) {
                    XYZ v1_value = v1.get_value();
                    XYZ v2_value = v2.get_value();
                    return v1_value[internal_index] < v2_value[internal_index];
                }
                return v1.midpoint[internal_index] < v2.midpoint[internal_index];
            }
        private:
            char internal_index;
            bool use_selector;
        };
    };
    pair<Split*, BinResults> binned_split_probe(vector<Tri*>* geo) {
        int min_bins = 10;
        int max_bins = 10;
        int bin_count = std::max(min_bins, std::min(max_bins, (int)geo->size()));
        int adaptive_bin_count = 3;
        int adpative_sweep_recusion_count = 1;

        int geo_count = geo->size();
        auto sorted = vector<relevant_value>();

        Split operator_split = Split(XYZ(), 0);
        Split best = operator_split;
        best.score = 999999999999999999999999.0;

        for (Tri* T_ptr : *geo) {
            sorted.push_back(relevant_value(T_ptr, 0));
            //sorted.push_back(relevant_value(T_ptr, 1));
            sorted.push_back(relevant_value(T_ptr, 2));
        }
        for (int i = 0; i < 3; i++) {
            operator_split.facing = i;
            sort(sorted.begin(), sorted.end(), relevant_value::comparer(i, true));

            XYZ left_min = sorted.front().min;
            XYZ left_max = sorted.front().min;
            XYZ right_min = sorted.back().min;
            XYZ right_max = sorted.back().max;
            int left_count = 0;
            int right_count = geo_count;



            for (int j = sorted.size() - 1; j >= 0; j--) {
                relevant_value& v = sorted[j];
                v.right_min = right_min;
                v.right_max = right_max;
                if (v.value_selector == 0) {
                    right_min = XYZ::min(right_min, v.min);
                    right_max = XYZ::max(right_max, v.max);
                }
            }

            double span = right_max[i] - right_min[i];
            double bin_interval = span / (bin_count);
            vector<double> bin_scores;

            operator_split.n.min = XYZ();
            operator_split.n.max = XYZ();
            operator_split.n.count = 0;
            operator_split.p.min = right_min;
            operator_split.p.max = right_max;
            operator_split.p.count = geo_count;
            evaluate_split(operator_split, 1);
            if (operator_split.score < best.score) best = operator_split;

            double pos = left_min[i];
            int index = 0;
            set<Tri*> partials;
            vector<vector<int>> debug;
            for (int j = 0; j < bin_count; j++) {
                while (index < sorted.size() && sorted[index].get_value()[i] < pos) {
                    relevant_value& v = sorted[index];
                    assert(v.parent != nullptr);
                    if (v.value_selector == 0) {
                        partials.insert(v.parent);
                        left_count++;
                    }
                    if (v.value_selector == 2) {
                        partials.erase(v.parent);
                        right_min = v.right_min;
                        right_max = v.right_max;
                        left_min = XYZ::min(v.min, left_min);
                        left_max = XYZ::max(v.max, left_max);
                        right_count--;
                    }
                    //debug.push_back(vector<int>());
                    //debug.back().push_back(left_count);
                    //debug.back().push_back(right_count);
                    //debug.back().push_back(index);
                    index++;
                }

                XYZ temp_right_min = right_min;
                XYZ temp_right_max = right_max;
                XYZ temp_left_min = left_min;
                XYZ temp_left_max = left_max;

                vector<XYZ> planar;
                for (Tri* T : partials) {
                    vector<XYZ> points;
                    vector<XYZ> left; //note, this is probably pretty inefficient, but its a simple solution
                    vector<XYZ> right;
                    points.push_back(T->p1);
                    points.push_back(T->p2);
                    points.push_back(T->p3);
                    for (int ti = 0; ti < 3; ti++) {
                        XYZ p = points[ti];
                        if (p[i] < pos)left.push_back(p);
                        if (p[i] == pos)planar.push_back(p);
                        if (p[i] > pos)right.push_back(p);
                    }

                    for (int li = 0; li < left.size(); li++) {
                        for (int ri = 0; ri < right.size(); ri++) {
                            XYZ v1 = left[li];
                            XYZ v2 = right[ri];
                            XYZ slope = XYZ::slope(v1, v2);
                            double t_span = v2[i] - v1[i];
                            double t = (pos - v1[i]) / t_span;
                            XYZ planar_point = slope * t + v1;
                            planar.push_back(planar_point);
                        }
                    }
                }
                for (XYZ& p : planar) {
                    temp_right_min = XYZ::min(p, temp_right_min);
                    temp_right_max = XYZ::max(p, temp_right_max);
                    temp_left_min = XYZ::min(p, temp_left_min);
                    temp_left_max = XYZ::max(p, temp_left_max);
                }
                operator_split.n.min = temp_left_min;
                operator_split.n.max = temp_left_max;
                operator_split.n.count = left_count;
                operator_split.p.min = temp_right_min;
                operator_split.p.max = temp_right_max;
                operator_split.p.count = right_count;
                operator_split.placement = temp_left_max;
                evaluate_split(operator_split, 1);
                if (operator_split.score < 0) {
                    assert(0);
                }
                if (operator_split.score < best.score) {
                    best = operator_split;
                }
                bin_scores.push_back(operator_split.score);
                pos += bin_interval;

            }
            for (int i = 0; i < bin_scores.size(); i++) {
                cout << bin_scores[i] << endl;
            }
        }
        exit(0);

        BinResults out_bin = BinResults();
        out_bin.n_geo = new vector<Tri*>();
        out_bin.p_geo = new vector<Tri*>();
        int facing = best.facing;
        for (Tri* T : *geo) {
            if (T->AABB_max[facing] < best.placement[facing]) {
                out_bin.n_geo->push_back(T);
                continue;
            }
            if (T->AABB_min[facing] > best.placement[facing]) {
                out_bin.p_geo->push_back(T);
                continue;
            }

            vector<XYZ> points;
            vector<XYZ> left; //note, this is probably pretty inefficient, but its a simple solution
            vector<XYZ> right;
            points.push_back(T->p1);
            points.push_back(T->p2);
            points.push_back(T->p3);
            Tri* left_T = new Tri(*T);
            Tri* right_T = new Tri(*T);
            left_T->AABB_min = T->AABB_max;
            left_T->AABB_max = T->AABB_min;
            right_T->AABB_min = T->AABB_max;
            right_T->AABB_max = T->AABB_min;
            double pos = best.placement[facing];
            for (int ti = 0; ti < 3; ti++) {
                XYZ p = points[ti];
                if (p[facing] < pos) {
                    left.push_back(p);
                    left_T->AABB_min = XYZ::min(left_T->AABB_min, p);
                    left_T->AABB_max = XYZ::max(left_T->AABB_max, p);
                }
                if (p[facing] == pos) {
                    left_T->AABB_min = XYZ::min(left_T->AABB_min, p);
                    left_T->AABB_max = XYZ::max(left_T->AABB_max, p);
                    right_T->AABB_min = XYZ::min(right_T->AABB_min, p);
                    right_T->AABB_max = XYZ::max(right_T->AABB_max, p);
                }
                if (p[facing] > pos) {
                    right_T->AABB_min = XYZ::min(right_T->AABB_min, p);
                    right_T->AABB_max = XYZ::max(right_T->AABB_max, p);
                }
            }

            for (int li = 0; li < left.size(); li++) {
                for (int ri = 0; ri < right.size(); ri++) {
                    XYZ v1 = left[li];
                    XYZ v2 = right[ri];
                    XYZ slope = XYZ::slope(v1, v2);
                    double t_span = v2[facing] - v1[facing];
                    double t = (pos - v1[facing]) / t_span;
                    XYZ p = slope * t + v1;
                    left_T->AABB_min = XYZ::min(left_T->AABB_min, p);
                    left_T->AABB_max = XYZ::max(left_T->AABB_max, p);
                    right_T->AABB_min = XYZ::min(right_T->AABB_min, p);
                    right_T->AABB_max = XYZ::max(right_T->AABB_max, p);
                }
            }
            out_bin.n_geo->push_back(left_T);
            out_bin.p_geo->push_back(right_T);
        }
        return pair<Split*, BinResults>(new Split(best), out_bin);
    }
    Split* probe(vector<Tri*>* geo) {

        auto sorted = vector<relevant_value>();
        for (Tri* T_ptr : *geo) {
            sorted.push_back(relevant_value(T_ptr));
        }
        Split operator_split = Split(XYZ(), 0);
        Split best = operator_split;
        best.score = 999999999999999999999999.0;
        int geo_count = geo->size();
        int iteration = 0;
        double last_read = 0;
        for (int i = 0; i < 3; i++) {
            operator_split.facing = i;
            sort(sorted.begin(), sorted.end(), relevant_value::comparer(i));

            XYZ left_min = sorted.front().min;
            XYZ left_max = sorted.front().max;
            XYZ right_min = sorted.back().min;
            XYZ right_max = sorted.back().max;
            int count = 0;
            for (int j = sorted.size() - 1; j >= 0; j--) {
                relevant_value& v = sorted[j];
                right_min = XYZ::min(right_min, v.min);
                right_max = XYZ::max(right_max, v.max);
                v.right_min = right_min;
                v.right_max = right_max;
            }

            operator_split.n.min = XYZ();
            operator_split.n.max = XYZ();
            operator_split.n.count = 0;
            operator_split.p.min = right_min;
            operator_split.p.max = right_max;
            operator_split.p.count = geo_count;
            evaluate_split(operator_split, 1);
            if (operator_split.score < best.score) best = operator_split;

            double prev_pos = -99999999;
            int segments = 100;
            double span = right_max[i] - right_min[i];
            double segment = segments / span;


            for (int j = 0; j < sorted.size(); j++) {
                auto v = sorted[j];
                left_min = XYZ::min(left_min, v.min);
                left_max = XYZ::max(left_max, v.max);
                count++;
                prev_pos = v.midpoint[i];
                operator_split.n.min = left_min;
                operator_split.n.max = left_max;
                operator_split.n.count = count;
                operator_split.p.min = v.right_min;
                operator_split.p.max = v.right_max;
                operator_split.p.count = geo_count - count;
                operator_split.placement = v.midpoint;
                evaluate_split(operator_split, 1);
                if (operator_split.score < best.score) {
                    best = operator_split;
                }
                if (abs(operator_split.score - last_read) / operator_split.score > 0.05) {
                    //cout << iteration << endl;
                    //cout << operator_split.score << endl;
                    last_read = operator_split.score;
                }
                iteration++;
            }
        }
        //exit(0);

        return new Split(best);
    }
    static void operate_stats(Stats& stat, const XYZ& point_max, const XYZ& point_min) {
        stat.count++;
        stat.max = XYZ::max(point_max, stat.max);
        stat.min = XYZ::min(point_min, stat.min);
    }
    static void get_stats(vector<Tri*>* geo, Split& split) {
        split.p.count = 0;
        split.n.count = 0;
        split.p.max = XYZ(-99999999);
        split.n.max = XYZ(-99999999);
        split.p.min = XYZ(9999999);
        split.n.min = XYZ(9999999);
        int axis = split.facing;
        for (const Tri* T_ptr : *geo) {
            if (T_ptr->midpoint[axis] > split.placement[axis]) {
                operate_stats(split.p, T_ptr->AABB_max, T_ptr->AABB_min);
            }
            if (T_ptr->midpoint[axis] <= split.placement[axis]) {
                operate_stats(split.n, T_ptr->AABB_max, T_ptr->AABB_min);
            }
        }
    }
    static decimal evaluate_split(Split& split, decimal SA_parent) {
        decimal traversal_cost = 1;
        decimal intersect_cost = 1;
        decimal Sa = split.p.SA() / SA_parent;
        decimal Sb = split.n.SA() / SA_parent;
        int count_a = split.p.count;
        int count_b = split.n.count;
#if PENALIZE_UNFILLED_LEAFS
        count_a = ceil(count_a / (float)LEAF_SIZE);
        count_b = ceil(count_b / (float)LEAF_SIZE);
        intersect_cost = 2;
#endif
        decimal Ha = Sa * count_a * intersect_cost;
        decimal Hb = Sb * count_b * intersect_cost;
        decimal final_h = traversal_cost + Ha + Hb;
        split.score = final_h;
        return final_h;
        //return (split.p.count+split.n.count)/geo->size()*(split.p.count+split.n.count);
    }

};



class PackagedBVH { //memory optimized. produced once the BVH tree is finalized
public:
    int32_t index;
    XYZ sMax;
    XYZ sMin;
    int32_t leaf_size;
    PackagedBVH(BVH* target) {
        sMax = target->max;
        sMin = target->min;
        index = -1;
    }
    static pair<PackagedBVH*, vector<PackagedTri>*> collapse(BVH* top) {
        vector<PackagedTri>* tri_vec = new vector<PackagedTri>();
        int tri_count = top->count();
        int tree_size = top->size();
        PackagedBVH* out = (PackagedBVH*)_aligned_malloc((tree_size + 1) * sizeof(PackagedBVH), 32);
        tri_vec->reserve(tri_count + 1);
        PackagedBVH::collapse(out, tri_vec, top);
        return pair<PackagedBVH*, vector<PackagedTri>*>(out, tri_vec);
    }
private:
    static void collapse(PackagedBVH* vec, vector<PackagedTri>* tri_vec, BVH* top) {
        vector<pair<BVH*, int>> current;
        vector<pair<BVH*, int>> next;

        int accumulated_index = 0;
        current.push_back(pair<BVH*, int>(top, -1));
        while (current.size() > 0) {
            for (int i = 0; i < current.size(); i++) {
                auto selection = current.at(i);
                BVH* target = selection.first;
                int parent_index = selection.second;
                auto PBVH = PackagedBVH(target);
                PBVH.leaf_size = target->elements.size();
                if (PBVH.leaf_size > 0) {
                    PBVH.index = tri_vec->size();
                    for (int i = 0; i < PBVH.leaf_size; i++) {
                        tri_vec->push_back(target->elements[i]->pack());
                    }
                }
                if (target->c1 != nullptr) {
                    next.push_back(pair<BVH*, int>(target->c1, accumulated_index));
                    next.push_back(pair<BVH*, int>(target->c2, accumulated_index));
                }
                vec[accumulated_index] = PBVH;
                if (parent_index != -1) {
                    if (vec[parent_index].index == -1) {
                        vec[parent_index].index = accumulated_index;
                    }

                }
                else accumulated_index++;
                accumulated_index++;
            }
            current = next;
            next.clear();
        }
    }
};



int total_tris = 0;
int nodes_traversed = 0;


struct PackagedRay {
    XYZ position;
    XYZ slope;
    char generation = 0;
    XYZ* output;
    XYZ coefficient;
    PackagedRay() {};
    PackagedRay(XYZ _position, XYZ _slope, char gen) :
        position(_position),
        slope(_slope),
        generation(gen)
    {}
    void move(decimal distance) {
        position += slope * distance;
    }
};

struct MinimumRay {
    XYZ position;
    XYZ inv_slope;
    uint64_t index;
};

class BVH_AVX {
public:
    m256_vec3 max;
    m256_vec3 min;
    unsigned int leaf_size[8];
    unsigned int indexes[8];
    int parent_index;
    int self_index;
    vector<MinimumRay>* pool;
    BVH_AVX(vector<BVH*>& batch) {
        vector<XYZ> max_vec;
        vector<XYZ> min_vec;
        for (int i = 0; i < batch.size(); i++) {
            max_vec.push_back(batch[i]->max);
            min_vec.push_back(batch[i]->min);
            leaf_size[i] = batch[i]->elements.size();
            total_tris += leaf_size[i];
        }
        for (int i = batch.size(); i < 8;i++) {
            max_vec.push_back(XYZ());
            min_vec.push_back(XYZ());
            leaf_size[i] = 0;
            indexes[i] = 0;
        }
        max = m256_vec3(max_vec);
        min = m256_vec3(min_vec);

        pool = new vector<MinimumRay>();

    }
    __m256 intersectionv1(const m256_vec3& fusedorigin, const m256_vec3& inv_slope) const { //holy mother of moving data. I pray for you, my cpu, I pray
        __m256 tx1 = _mm256_fmadd_ps(min.X, inv_slope.X, fusedorigin.X);
        __m256 tx2 = _mm256_fmadd_ps(max.X, inv_slope.X, fusedorigin.X);
        __m256 ty1 = _mm256_fmadd_ps(min.Y, inv_slope.Y, fusedorigin.Y);
        __m256 ty2 = _mm256_fmadd_ps(max.Y, inv_slope.Y, fusedorigin.Y);
        __m256 tz1 = _mm256_fmadd_ps(min.Z, inv_slope.Z, fusedorigin.Z);
        __m256 tz2 = _mm256_fmadd_ps(max.Z, inv_slope.Z, fusedorigin.Z);

        __m256 tmin = _mm256_max_ps(
            _mm256_min_ps(tz1, tz2),
            _mm256_max_ps(
                _mm256_min_ps(ty1, ty2),
                _mm256_min_ps(tx1, tx2)
            ));
        __m256 tmax = _mm256_min_ps(
            _mm256_max_ps(tz1, tz2),
            _mm256_min_ps(
                _mm256_max_ps(ty1, ty2),
                _mm256_max_ps(tx1, tx2)
            ));
        __m256 diff = _mm256_sub_ps(tmax, tmin);
        __m256 masked_diff = _mm256_max_ps(
            diff,
            AVX_ZEROS
        );
        __m256 return_mask = _mm256_cmp_ps(
            masked_diff,
            diff,
            _CMP_EQ_OQ
        );
        __m256 final = _mm256_and_ps(
            _mm256_add_ps(
                tmin,
                masked_diff
            ),
            return_mask
        );
        //__m256 diff_is_pos = _mm256_cmp_ps(diff, AVX_ZEROS, _CMP_GT_OQ);

        return final;
    }
    __m256 intersection(const m256_vec3& fusedorigin, const m256_vec3& inv_slope) const { //holy mother of moving data. I pray for you, my cpu, I pray
        __m256 tx1 = _mm256_fmadd_ps(min.X, inv_slope.X, fusedorigin.X);
        __m256 tx2 = _mm256_fmadd_ps(max.X, inv_slope.X, fusedorigin.X);
        __m256 ty1 = _mm256_fmadd_ps(min.Y, inv_slope.Y, fusedorigin.Y);
        __m256 ty2 = _mm256_fmadd_ps(max.Y, inv_slope.Y, fusedorigin.Y);
        __m256 tz1 = _mm256_fmadd_ps(min.Z, inv_slope.Z, fusedorigin.Z);
        __m256 tz2 = _mm256_fmadd_ps(max.Z, inv_slope.Z, fusedorigin.Z);

        __m256 tminx = _mm256_min_ps(tx1, tx2);
        __m256 tminy = _mm256_min_ps(ty1, ty2);
        __m256 tminz = _mm256_min_ps(tz1, tz2);
        __m256 tmaxx = _mm256_max_ps(tx1, tx2);
        __m256 tmaxy = _mm256_max_ps(ty1, ty2);
        __m256 tmaxz = _mm256_max_ps(tz1, tz2);

        __m256 tmin = _mm256_max_ps(
            tminz,
            _mm256_max_ps(
                tminy,
                tminx
            ));
        __m256 tmax = _mm256_min_ps(
            tmaxz,
            _mm256_min_ps(
                tmaxy,
                tmaxx
            ));
        __m256 diff = _mm256_sub_ps(tmin, tmax);
        __m256 return_mask = _mm256_cmp_ps(
            tmin,
            tmax,
            _CMP_LT_OQ
        );
        __m256 min_mask = _mm256_cmp_ps(
            tmin,
            AVX_ZEROS,
            _CMP_GE_OQ
        );
        __m256 final = _mm256_and_ps(
            _mm256_add_ps(
                tmax,
                _mm256_and_ps(
                    diff,
                    min_mask
                )
            ),
            return_mask
        );
        //__m256 diff_is_pos = _mm256_cmp_ps(diff, AVX_ZEROS, _CMP_GT_OQ);

        return final;
    }
    static pair<vector<BVH_AVX>*, vector<PackagedTri>*> collapse(BVH* top) {
        vector<PackagedTri>* tri_vec = new vector<PackagedTri>();
        int tri_count = top->count();
        int tree_size = top->size();
        vector<BVH_AVX> v;
        v.reserve(top->size());
        tri_vec->reserve(tri_count + 1);
        BVH_AVX::collapse(v, tri_vec, top);
        vector<BVH_AVX>* out = new vector<BVH_AVX>();
        out->swap(v);//trimming overallocation of vector
        return pair<vector<BVH_AVX>*, vector<PackagedTri>*>(out, tri_vec);
    }
private:
    struct comparer {
        bool operator()(BVH* b1, BVH* b2) {
            //return b1->avg_path() < b2->avg_path();
            return VecLib::surface_area(b1->max, b1->min) * b1->count() > VecLib::surface_area(b2->max, b2->min) * b2->count();
        }
    };
    static void collapse(vector<BVH_AVX>& vec, vector<PackagedTri>* tri_vec, BVH* top) {
        vector<pair<BVH*, pair<int, int>>> current;
        vector<pair<BVH*, pair<int, int>>> next;

        int accumulated_index = 0;
        current.push_back(pair<BVH*, pair<int, int>>(top, pair<int, int>(-1, -1)));
        while (current.size() > 0) {
            for (int i = 0; i < current.size(); i++) {
                auto selection = current.at(i);
                vector<BVH*> batch;
                vector<BVH*> has_tris;
                batch.push_back(selection.first);
                while (true) {
                    while (batch.size() + has_tris.size() < 8) {
                        BVH* at_index = batch[0];
                        if (at_index->c1 != nullptr) {
                            batch.push_back(at_index->c1);
                            batch.push_back(at_index->c2);
                            nodes_traversed += 2;
                            batch.erase(batch.begin());
                        }
                        else {
                            has_tris.push_back(at_index);
                            batch.erase(batch.begin());
                            if (batch.size() == 0) {
                                break;
                            }
                        }
                        sort(batch.begin(), batch.end(), comparer());
                    }
                    if (has_tris.size() + batch.size() >= 8 || batch.size() == 0) {
                        if (has_tris.size() + batch.size() > 8) {
                            cout << "error occured in AVX BVH construction: over allocated" << endl;
                            throw exception();
                        }
                        for (int i = 0; i < has_tris.size(); i++) {
                            batch.push_back(has_tris[i]);
                        }
                        break;
                    }
                }
                BVH* target = selection.first;
                int parent_index = selection.second.first;
                int point_back_index = selection.second.second;
                BVH_AVX ABVH(batch);
                ABVH.parent_index = parent_index;
                ABVH.self_index = point_back_index;
                for (int i = 0; i < batch.size(); i++) {
                    if (ABVH.leaf_size[i] > 0) {
                        ABVH.indexes[i] = tri_vec->size();
                        for (int j = 0; j < ABVH.leaf_size[i]; j++) {
                            tri_vec->push_back(batch[i]->elements[j]->pack());
                        }
                    }
                    else {
                        next.push_back(pair<BVH*, pair<int, int>>(batch[i], pair<int, int>(accumulated_index, i)));
                    }
                }
                vec.push_back(ABVH);
                if (parent_index != -1) {
                    BVH_AVX& parent = vec[parent_index];
                    for (int k = 0; k < 8; k++) {
                        parent.indexes[point_back_index] = accumulated_index;
                    }
                }
                accumulated_index++;
            }
            current = next;
            next.clear();
        }
    }
};



class Lens {
public:
    int resolution_x;
    int resolution_y;
    int subdivision_size;
    void (*_prep)(Lens* self, int resolution_x, int resolution_y, int subdivision_size);
    Lens* (*_clone)(Lens* self);
    XYZ(*_at)(Lens* self, int p_x, int p_y);
    void prep(int resolution_x, int resolution_y, int subdivision_size) {
        _prep(this, resolution_x, resolution_y, subdivision_size);
    }
    XYZ at(int p_x, int p_y) {
        return _at(this, p_x, p_y);
    }
    Lens* clone() {
        Lens* obj = _clone(this);
        obj->resolution_x = resolution_x;
        obj->resolution_y = resolution_y;
        obj->subdivision_size = subdivision_size;
        return obj;
    }

};

 class RectLens : public Lens {
public:
    decimal width;
    decimal height;
    vector<XY> subdiv_offsets;
    vector<vector<XY>> outputs;
    float pixel_width;
    float pixel_height;
    RectLens(decimal _width, decimal _height) :
        width(_width), height(_height) {
        _at = _at_function;
        _prep = _prep_function;
        _clone = _clone_function;
    }
private:
    static void _prep_function(Lens* self, int resolution_x, int resolution_y, int subdivision_count) {
        RectLens* self_rect = (RectLens*)self;
        decimal width = self_rect->width;
        decimal height = self_rect->height;
        decimal partial_width = width / resolution_x;
        decimal partial_height = height / resolution_y;
        self_rect->pixel_width = partial_width;
        self_rect->pixel_height = partial_height;
        decimal offset_x = -width / 2;
        decimal offset_y = -height / 2;
        for (int y = 0; y < resolution_y; y++) {
            vector<XY> row;
            for (int x = 0; x < resolution_x; x++) {
                decimal final_x = offset_x + partial_width * x;
                decimal final_y = offset_y + partial_height * y;
                row.push_back(XY(
                    final_x+partial_width/2,
                    final_y+partial_height/2
                ));
            }
            self_rect->outputs.push_back(row);
        }

    }
    static XYZ _at_function(Lens* self, int p_x, int p_y) {
        RectLens* self_rect = (RectLens*)self;
        XY pos = self_rect->outputs[p_y][p_x];
        return XYZ(pos.X, 0, pos.Y);
    }
    static Lens* _clone_function(Lens* self) {
        RectLens* rect_self = (RectLens*)self;
        Lens* obj = new RectLens(rect_self->width, rect_self->height);
        RectLens* rect_obj = (RectLens*)obj;
        rect_obj->outputs = rect_self->outputs;
        rect_obj->subdiv_offsets = rect_self->subdiv_offsets;
        return obj;
    }
};

 class Camera {
    //Ill note this is a fairly non-direct class. the original was more streamlined, but I decided to update it to remove gunk
    //and to allow progressive output updates during monte carlo rendering
public:
    XYZ position;
    Quat rotation = Quat(0, 0, 0, 1);

    Lens* lens;
    XYZ focal_position;

    int current_resolution_x = 0;
    int current_resolution_y = 0;

    Camera(XYZ _position, Lens* _lens, decimal focal_distance) :
        lens(_lens), position(_position), focal_position(XYZ(0, 0, -focal_distance)) {}

    Camera(XYZ _position, Lens* _lens, XYZ _focal_position) :
        lens(_lens), position(_position), focal_position(_focal_position) {}

    void prep(int resolution_x, int resolution_y, int samples) {
        current_resolution_x = resolution_x;
        current_resolution_y = resolution_y;
        lens->prep(resolution_x, resolution_y, samples);
    }

    XYZ slope_at(int p_x, int p_y) {
        return Quat::applyRotation(XYZ::slope(XYZ(0, 0, 0), lens->at(p_x, p_y) + focal_position), rotation);
    }


    Camera* clone() {
        Lens* new_lens = lens->clone();
        Camera* obj = new Camera(position, new_lens, focal_position);
        obj->current_resolution_x = current_resolution_x;
        obj->current_resolution_y = current_resolution_y;
        return obj;
    }

    ~Camera() {
        delete lens;
    }
};

const Matrix3x3 ACESOutputMat = Matrix3x3(
    1.60475, -0.53108, -0.07367,
    -0.10208, 1.10813, -0.00605,
    -0.00327, -0.07276, 1.07602
);

const Matrix3x3 ACESInputMat = Matrix3x3( //https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

struct CastResults {
    static const decimal default_distance;
    XYZ normal;
    Material* material;
    decimal distance;
    XYZ UV = XYZ(-1);
    int parent_index = 0;
    CastResults() : normal(XYZ()), material(nullptr), distance(default_distance) {};
    CastResults(XYZ _normal, Material* _mat) : normal(_normal), material(_mat), distance(default_distance) {}
};

const decimal CastResults::default_distance = 99999999999999999;

struct Casting_Diagnostics {
    long reflections_cast = 0;
    long shadows_cast = 0;
    long diffuses_cast = 0;
    long long rays_processed = 0;
    chrono::nanoseconds duration;
};

class PackagedScene {
public:
    vector<PackagedTri> tri_data;
    vector<PTri_AVX> avx_tri_data;
    vector<PointLikeLight> lights;
    vector<PackagedTri> emissive_tris;
    PackagedBVH* flat_bvh = nullptr;
    vector<BVH_AVX> avx_bvh;
    short monte_carlo_generations = 2;
    short max_generations = 3;
    short monte_carlo_max = 64;
    float monte_carlo_modifier = 1.0 / 32;
};

 class Scene {
public:

    int object_count = 0;
    int primitive_count = 0;

    vector<Object*> objects;
    vector<Camera*> cameras;
    Camera* camera;

    vector<PointLikeLight*> pointlike_lights;

    int current_resolution_x;
    int current_resolution_y;
    Scene() {}

    void register_object(Object* obj) {
        object_count++;
        for (Mesh* M : obj->meshes) {
            primitive_count += M->primitive_count;
        }
        objects.push_back(obj);
    }
    void register_camera(Camera* cam) {
        cameras.push_back(cam);
    }
    void merge(Scene& mergee) {
        for (Object* O : mergee.objects) {
            objects.push_back(O);
        }
        for (Camera* cam : mergee.cameras) {
            cameras.push_back(cam);
        }
    }

    void prep(int res_x, int res_y, int subdiv_count) {
        auto start = chrono::high_resolution_clock::now();
        for (Object* O : objects) {
            O->prep();
        }
        camera = cameras[0];
        auto end = chrono::high_resolution_clock::now();
    }

    PackagedScene* package() {
        BVH* bvh = new BVH();


        PackagedScene* PS = new PackagedScene();

        auto data_group_start = chrono::high_resolution_clock::now();
        vector<Tri> tris;
        objects[0]->fetchData(tris);

        auto data_group_end = chrono::high_resolution_clock::now();

        //for (int i = 0; i < tris.size();i++) {
        //    Tri T = tris[i];
        //    cout << "triangle(" << T.p1.to_string() << "," << T.p2.to_string() << "," << T.p3.to_string() << ")" << endl;
        //}

        if (tris.size() > 0) {
            auto BVH_con_start = chrono::high_resolution_clock::now();

            bvh->construct_dangerous(tris); //note, this will cause the BVH to break when tris moves out of scope

            auto BVH_con_end = chrono::high_resolution_clock::now();
            auto BVH_collapse_start = chrono::high_resolution_clock::now();

            PS->emissive_tris = bvh->get_emissive_tris();
#if USE_AVX_BVH
            auto collapse_out = BVH_AVX::collapse(bvh);
            PS->avx_bvh = *collapse_out.first;
            PS->tri_data = *collapse_out.second;


            delete bvh;
#endif
#if USE_AVX_TRI
            map<int, pair<BVH_AVX*, int>> order;
            for (int i = 0; i < PS->avx_bvh.size(); i++) {
                BVH_AVX& current = PS->avx_bvh[i];
                for (int i = 0; i < 8; i++) {
                    int leaf_size = current.leaf_size[i];
                    int index = current.indexes[i];
                    if (leaf_size > 0) {
                        order[index] = pair<BVH_AVX*, int>(&current, i);
                    }
                }
            }
            for (auto iter = order.begin(); iter != order.end(); ++iter)
            {
                int index = iter->first;
                auto datapoint = iter->second;
                BVH_AVX& current = *datapoint.first;
                int selection_index = datapoint.second;
                int leaf_size = current.leaf_size[selection_index];
                current.indexes[selection_index] = PS->avx_tri_data.size();
                current.leaf_size[selection_index] = ceil(((float)leaf_size) / 8.0);
                int i = 0;
                while (i < ceil(((float)leaf_size) / 8.0)) {
                    vector<PackagedTri> PTris;
                    for (int j = 0; j < 8 && i * 8 + j < leaf_size; j++) {
                        PTris.push_back(PS->tri_data[index + 8 * i + j]);
                    }
                    PS->avx_tri_data.push_back(PTri_AVX(PTris));
                    i++;
                }
            }
#endif
            auto BVH_collapse_end = chrono::high_resolution_clock::now();
        }
        return PS;
    }
};
#define AVX_STACK_SIZE 256

#define USE_FIRE 1

class RayEngine {
public:
    PackagedScene* data;
    vector<vector<PackagedRay>> ray_packet_queue;
    const int max_queue = pow(2, 20); //bit over a million
    const int sub_packet_size = 1024;
    RayEngine() {}
    void load_scene(PackagedScene* PS) {
        data = PS;
    }
    void iterativeBVH_AVX(CastResults& res, const XYZ& position, const XYZ& slope) {
        //global_counter++;
        m256_vec3 position_avx(position);
        m256_vec3 slope_avx(slope);
        m256_vec3 fusedposition(-1 * position / slope);
        m256_vec3 inv_slope(1 / slope);
        int stack[AVX_STACK_SIZE];
        float distances[AVX_STACK_SIZE];
        int tri_count[AVX_STACK_SIZE];
        int stack_index = 0;
        stack[0] = 0;
        distances[0] = 0;
        tri_count[0] = 0;
        while (stack_index >= 0) {
            const float dist = distances[stack_index];
            const int self_triangles = tri_count[stack_index];
            if (dist >= res.distance) {
                stack_index--;
                continue;
            };
            if (self_triangles == 0) {
                const BVH_AVX& current = data->avx_bvh[stack[stack_index]];
                stack_index--;
                auto m256_results = current.intersection(fusedposition, inv_slope);
                //float results[8];
                float* results = (float*)&m256_results;
                int sorted_index[8];
                int sorted_index_size = 0;
                for (int i = 0; i < 8; i++) {
                    float& num = results[i];
                    if (num > 0) {
                        int j = sorted_index_size - 1;
                        for (; j >= 0; j--) {
                            if (results[sorted_index[j]] > num) {
                                sorted_index[j + 1] = sorted_index[j];
                            }
                            else break;
                        }
                        sorted_index[j + 1] = i;
                        sorted_index_size++;
                    }
                }
                for (int i = sorted_index_size - 1; i >= 0; i--) {
                    int selection_index = sorted_index[i];
                    float dist2 = results[selection_index];
                    if (dist2 <= 0) {
                        break;
                    }
                    stack_index++;
                    stack[stack_index] = current.indexes[selection_index];
                    distances[stack_index] = dist2;
                    tri_count[stack_index] = current.leaf_size[selection_index];

                }

            }
            else {
                int start_index = stack[stack_index];
                stack_index--;
                for (int i = 0; i < self_triangles; i++) {
#if !USE_AVX_TRI

                    const PackagedTri& t = (data->tri_data)[i + start_index];
                    XYZ distance = Tri::intersection_check_MoTru(t, position, slope);
                    if (distance.X >= 0) {
                        if (distance.X < res.distance) {
                            res.distance = distance.X;
                            res.normal = Tri::get_normal(t.normal, position + distance * slope * 0.99 - t.origin_offset);
                            res.material = t.material;
                            res.UV = distance;
                        }
                    }
#else

                    const PTri_AVX& PTri_AVX_pack = (data->avx_tri_data)[i + start_index];
                    m256_vec3 intersection_stats;
                    m256_vec3 normal_stats;
                    PTri_AVX::intersection_check(PTri_AVX_pack, position_avx, slope_avx, intersection_stats);
                    PTri_AVX::get_normal(PTri_AVX_pack, position_avx, normal_stats);

                    float* dists = (float*)&intersection_stats.X;
                    for (int i = 0; i < 8; i++) {
                        if (dists[i] > 0 && dists[i] < res.distance) {
                            res.distance = dists[i];
                            res.material = (Material*)102394123;
                            res.normal = normal_stats.at(i);
                            res.UV = intersection_stats.at(i);
                        }
                    }
#endif
                }

            }
        }
    }
    void navBVH(CastResults& res, const XYZ& position, const XYZ& slope, const XYZ& inv_slope) {
        iterativeBVH_AVX(res, position, slope);
    }
    CastResults execute_bvh_cast(XYZ& position, const XYZ& slope) {
        CastResults returner;
        XYZ inv_slope = XYZ(1) / slope;
        navBVH(returner, position, slope, inv_slope);
        position += returner.distance * slope * 0.999;
        return returner;
    }
    CastResults execute_naive_cast(XYZ& position, const XYZ& slope) {
#define default_smallest_distance 9999999;
        decimal smallest_distance = default_smallest_distance;
        CastResults returner = CastResults(XYZ(-1, -1, -1), nullptr);
        for (const PackagedTri& t : data->tri_data) {
            XYZ distance = Tri::intersection_check(t, position, slope);
            if (distance.X >= 0) {
                if (distance.X < returner.distance) {
                    returner.distance = distance.X;
                    returner.material = (Material*)102394123;
                    returner.normal = Tri::get_normal(t.normal, position + distance * slope * 0.99 - t.origin_offset);
                }
            }
        }
        position += returner.distance * slope;
        return returner;
    }
    CastResults execute_ray_cast(XYZ& position, const XYZ& slope) {
        return execute_bvh_cast(position, slope);
        //return execute_naive_cast(position,slope);

    }
    XYZ dispatch_ray(PackagedRay& ray_data) {

    }
    XYZ process_ray(Casting_Diagnostics& stats, PackagedRay& ray_data) {
        stats.rays_processed++;
        XYZ start = ray_data.position;
        CastResults results = execute_ray_cast(ray_data.position, ray_data.slope);
        //*ray_data.output += ray_data.coefficient * ((results.normal)+1)/2;
        if (results.material == nullptr) {
            return XYZ(0);
        }
        XYZ aggregate = XYZ();
        XYZ normal = results.normal;

        if (DRAW_UV) {
            if (results.UV.X != -1) {
                XYZ color;
                color.X = results.UV.Y;
                color.Y = results.UV.Z;
                color.Z = 1 - color.X - color.Y;
                return color * 1;
            }
        }
        if (DRAW_NORMAL) {
            return (normal / 2 + XYZ(0.5, 0.5, 0.5)) * 1;
        }
      
        //if (DRAW_BOUNCE_DIRECTION) {
        //    if (ray_data.remaining_monte_carlo == 0)
        //        (*ray_data.output) += ray_data.coefficient * (ray_data.position / 2 + XYZ(0.5, 0.5, 0.5)) * 10;
        //    //continue;
        //}
        //if (ray_data.generation >= data->monte_carlo_max || ray_data.generation == 0) {
        aggregate = 0;
        //}

        auto r2 = PackagedRay(ray_data);
        r2.position += r2.slope * 0.01;
        CastResults res2 = execute_ray_cast(r2.position, r2.slope);
        float dist = r2.position.X / r2.slope.X - ray_data.position.X / ray_data.slope.X;
        float dist_1 = ray_data.position.X/r2.slope.X - start.X / r2.slope.X;
#if USE_FIRE
        return (1-dist_1/20)*XYZ::linear_mix(dist/1,XYZ(1,1,1),XYZ(1,0.5,0));
#else
        return 0.5*XYZ::linear_mix(dist / 1, XYZ(0, 0.1, 0.4), XYZ(0.0, 0.5, 0.5))+0.7*pow(XYZ::dot(normal,XYZ::normalize(XYZ(1,1,1))),2);
#endif

    }
};



class GUIHandler {
public:
    int current_resolution_x = 0;
    int current_resolution_y = 0;

    sf::RenderWindow* window;
    sf::Image canvas;
    sf::Texture render_texture;
    sf::Sprite render_sprite;

    map<string, sf::Texture> textures;
    map<string, sf::Sprite> sprites;
    map<string, sf::Image> images;
    map<string, sf::RectangleShape> rects;

    double scalar_exponent = 0;

    XY focus_size;
    vector<XY> focuses;

    GUIHandler() {}

    GUIHandler(int resX, int resY, int block_size) {
        current_resolution_x = resX;
        current_resolution_y = resY;
        canvas.create(current_resolution_x, current_resolution_y, sf::Color::Black);
        render_sprite.setScale(make_scale(), -make_scale());
        render_sprite.setPosition(0, make_scale() * current_resolution_y);
        focus_size = XY(block_size, block_size);
    }

    void hold_window() {
        auto frame_time = chrono::high_resolution_clock::now();
        while (window->isOpen())
        {
            sf::Event event;
            while (window->pollEvent(event))
            {
                handle_events(event);
            }
            partial_render();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
    XY get_relative_position(int x, int y) {
        double relative_x = ((double)x) / current_resolution_x;
        double relative_y = ((double)y) / current_resolution_y;
        return XY(make_scale() * relative_x, make_scale() * relative_y);

    }
    void create_window() {
        cout << "Opening Window...." << flush;
        window = new sf::RenderWindow(sf::VideoMode(current_resolution_x * make_scale(), current_resolution_y * make_scale()), "Render Window!");
        cout << "Done" << endl;
    }
    void commit_pixel(XYZ color, int x, int y) {
        auto sfcolor = sf::Color(color.X, color.Y, color.Z);
        canvas.setPixel(x, y, sfcolor);
    }
    void partial_render() {
        window->clear();
        draw_render_sprite();
        window->display();
    }
    void render_pass() {
        window->clear();
        update_focuses();
        draw_render_sprite();
        draw_secondary_sprites();
        draw_shapes();
        window->display();
    }
    void update_focuses() {
        XY scaled_size = focus_size * make_scale();
        for (int i = 0; i < focuses.size(); i++) {
            XY focus_position = focuses.at(i);
            focus_position.Y = current_resolution_y - focus_position.Y - focus_size.Y;
            XY scaled_position = focus_position * make_scale();
            string focus_id = "render_focus_" + to_string(i);
            rects[focus_id].setSize(sf::Vector2f(scaled_size.X, scaled_size.Y));
            rects[focus_id].setPosition(scaled_position.X, scaled_position.Y);
        }
    }
    void draw_render_sprite() {
        window->draw(render_sprite);
    }
    void draw_secondary_sprites() {
        for (auto i = sprites.begin(); i != sprites.end(); i++) {
            sf::Sprite& sprite = i->second;
            window->draw(sprite);

        }
    }
    void draw_shapes() {
        for (auto i = rects.begin(); i != rects.end(); i++) {
            sf::RectangleShape& rect = i->second;
            window->draw(rect);
        }
    }
    void update_image_textures() {
        for (auto i = images.begin(); i != images.end(); i++) {
            string key = i->first;
            sf::Image& image = i->second;
            getTexture(key).loadFromImage(image);
        }
    }
    void commit_canvas() {
        render_texture.loadFromImage(canvas);
        render_sprite.setTexture(render_texture, false);
        render_pass();
    }
    void add_texture_sprite(string name) {
        textures[name] = sf::Texture();
        sprites[name] = sf::Sprite();
        getSprite(name).setTexture(getTexture(name));
    }
    void add_texture_sprite_image(string name) {
        textures[name] = sf::Texture();
        sprites[name] = sf::Sprite();
        images[name] = sf::Image();
        getSprite(name).setTexture(getTexture(name));
    }
    sf::RectangleShape& add_rect(string name) {
        rects[name] = sf::RectangleShape();
        return rects[name];
    }
    void add_focus() {
        string focus_id = "render_focus_" + to_string(focuses.size());
        add_rect(focus_id);
        focuses.push_back(XY(0, 0));

        sf::RectangleShape& focus = rects[focus_id];
        focus.setFillColor(sf::Color::Transparent);
        focus.setOutlineThickness(1);
        focus.setOutlineColor(sf::Color(255, 211, 110));
    }
    void set_focus_position(int index, int x, int y) {
        focuses[index] = XY(x, y);
    }
    sf::Sprite& getSprite(string name) {
        return sprites[name];
    }
    sf::Texture& getTexture(string name) {
        return textures[name];
    }
    sf::Image& getImage(string name) {
        return images[name];
    }
    void handle_events() {
        sf::Event event;
        while (window->pollEvent(event)) {
            handle_events(event);
        }
    }
    double make_scale() {
        return pow(2, scalar_exponent);
    }
    void handle_events(sf::Event event) {
        if (event.type == sf::Event::Closed) window->close();
        if (event.type == sf::Event::Resized)
        {
            // update the view to the new size of the window
            sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
            window->setView(sf::View(visibleArea));
        }
        if (event.type == sf::Event::MouseWheelMoved)
        {
            // display number of ticks mouse wheel has moved
            double amount = event.mouseWheel.delta;
            scalar_exponent += amount / 5;
            double scale = make_scale();
            render_sprite.setScale(scale, -scale);
            render_sprite.setPosition(0, make_scale() * current_resolution_y);
        }
    }
};
namespace ImageHandler {
    OCIO::ConstConfigRcPtr config = OCIO::Config::CreateFromFile("C:\\Users\\Charlie\\Libraries\\OCIOConfigs\\AgX-main\\config.ocio");
    OCIO::ConstProcessorRcPtr processor = config->getProcessor(OCIO::ROLE_SCENE_LINEAR, OCIO::ROLE_COLOR_PICKING);
    auto compute = processor->getDefaultCPUProcessor();
    static decimal luminance(XYZ v)
    {
        return XYZ::dot(v, XYZ(0.2126, 0.7152, 0.0722));
    }

    static XYZ change_luminance(XYZ c_in, decimal l_out)
    {
        decimal l_in = luminance(c_in);
        return c_in * (l_out / l_in);
    }

    //decimal realistic_response(decimal f, decimal iso) // https://graphics-programming.org/resources/tonemapping/index.html

    static decimal Filmic_curve(decimal t) {
        return 0.371 * (sqrt(t) + 0.28257 * log(t) + 1.69542);
    }
    static XYZ reinhard_extended_luminance(XYZ v, decimal max_white_l)
    {
        decimal l_old = luminance(v);
        decimal numerator = l_old * (1.0 + (l_old / (max_white_l * max_white_l)));
        decimal l_new = numerator / (1.0 + l_old);
        return change_luminance(v, l_new);
    }


    static XYZ RRTAndODTFit(XYZ v)
    {
        XYZ a = v * (v + 0.0245786f) - 0.000090537f;
        XYZ b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
        return a / b;
    }
    static XYZ ACESFitted(XYZ color)
    {
        color = Matrix3x3::multiply_horizontally(ACESInputMat, color);

        // Apply RRT and ODT
        color = RRTAndODTFit(color);

        color = Matrix3x3::multiply_horizontally(ACESOutputMat, color);

        return XYZ::clamp(color, 0, 1) * 255;
    }
    static XYZ FastACES(XYZ x) {
        decimal a = 2.51;
        decimal b = 0.03;
        decimal c = 2.43;
        decimal d = 0.59;
        decimal e = 0.14;
        return XYZ::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1) * 255;
    }
    XYZ post_process_pixel(XYZ luminence, int pre_scaled_gain, int post_scaled_gain) {
        XYZ scaled_return = luminence * pre_scaled_gain;
        decimal scalar_value = scaled_return.magnitude();
        //scaled_return = scaled_return * Filmic_curve(scalar_value);
        //scaled_return = scaled_return*log10(luminance(scaled_return) + 1) / luminance(scaled_return) * post_scaled_gain;//maybe more correct? dunno.
        scaled_return = XYZ::log(scaled_return + 1) * post_scaled_gain;
        //scaled_return = ACESFitted(scaled_return);
        //scaled_return = FastACES(scaled_return);
        //scaled_return = reinhard_extended_luminance(scaled_return, 200);
        scaled_return = XYZ::clamp(scaled_return, 0, 1) * 255;
        float pixel[3] = { luminence[0],luminence[1],luminence[2] };
        return XYZ::clamp(XYZ(pixel[0], pixel[1], pixel[2]), 0, 1) * 255;//scaled_return;
    }
    static void postProcessRaw(vector<vector<XYZ*>>* data) {}
    static GUIHandler* openImageVector(vector<vector<XYZ*>>* data) {
        int resX = data->at(0).size();
        int resY = data->size();
        cout << resX << " " << resY << endl;
        GUIHandler* GUI = new GUIHandler(resX, resY, 0);
        for (int y = 0; y < resY;y++) {
            for (int x = 0; x < resX; x++) {
                GUI->commit_pixel(*(data->at(y)[x]), x, y);
            }
        }
        GUI->create_window();
        GUI->commit_canvas();
        return GUI;
    }
};

class RenderThread {
public:
    struct block {
        int size;
        int block_index_x;
        int block_index_y;
        iXY offset;
        vector<XYZ*> raws;
        atomic<int> pixels_done;
        atomic<int> iterations_done{ 0 };
        int pixels_last;
        atomic<bool> done = false;
        block(int _block_size, int _block_x, int _block_y) {
            size = _block_size;
            block_index_x = _block_x;
            block_index_y = _block_y;
            int x_offset = block_index_x * size;
            int y_offset = block_index_y * size;
            offset = iXY(x_offset, y_offset);
            pixels_done = 0;
            pixels_last = 0;
            for (int i = 0; i < size * size; i++) {
                raws.push_back(new XYZ());
            }
        }
    };
    struct processing_info {
        processing_info() {}

        Casting_Diagnostics CD;
        atomic<long long> parent_rays_cast{ 0 };
        atomic<long long> child_rays_cast{ 0 };

        int block_size = 0;
        int res_y = 0;
        int res_x = 0;
        int y_increment = 0;
        int x_increment = 0;
        int pixels_per_block = 0;
        int samples_per_pixel = 0;

        atomic<int> pixels_done{ 0 };
        atomic<int> iterations_done{ 0 };

        Camera* camera = nullptr;
        RayEngine* RE = nullptr;

        XYZ emit_coord;
        XYZ starting_coefficient;


        void add_casts() {
            parent_rays_cast++;
            child_rays_cast += CD.rays_processed - 1;
            CD.rays_processed = 0;
        }
        long long rays_cast() {
            return parent_rays_cast + child_rays_cast;
        }
        double percent() {
            return ((double)pixels_done) / (res_x * res_y);
        }

        ~processing_info() {
            //delete camera;
        }
    };
    atomic<bool> idle = true;
    atomic<bool> stop_thread = false;
    atomic<bool> halt_ops = false;
    atomic<int> mode = 0;

    thread worker;
    block* current;
    processing_info* process_info;
    vector<block*> blocks;

    RenderThread(processing_info* config) :process_info(config), worker(&RenderThread::main, this) {

    }

    void main() {
        srand(chrono::high_resolution_clock::now().time_since_epoch().count());
        gen.seed(rand(), rand(), rand(), rand());
        while (true) {
            if (stop_thread.load(std::memory_order_relaxed)) {
                return;
            }
            if (!idle.load(std::memory_order_relaxed)) {
                int block_id = 0;
                while (!halt_ops.load(std::memory_order_relaxed)) {
                    current = blocks[block_id];
                    process_iteratively();
                    block_id++;
                    if (block_id == blocks.size()) {
                        block_id = 0;
                        idle = true;
                        break;
                    }
                }
                current = nullptr;
            }
            this_thread::sleep_for(chrono::milliseconds(1));
        }
    }
    void process_iteratively() {
        for (int pixel_index = 0; pixel_index < process_info->pixels_per_block; pixel_index++) {
            iXY pixel_coordinate = get_coord(pixel_index);
            XYZ* output_link = get_output_link(pixel_index);
            XYZ ray_slope = slope_at(pixel_coordinate);
            auto ray = PackagedRay(
                process_info->emit_coord,
                ray_slope,
                0
            );
            (*output_link) = process_info->RE->process_ray(process_info->CD, ray);
        }
    }
    iXY get_coord(int pixel_index) {
        int block_size = process_info->block_size;
        iXY inside_offset = iXY(pixel_index % block_size, block_size - pixel_index / block_size - 1);
        return current->offset + inside_offset;
    }
    XYZ* get_output_link(int pixel_index) {
        return current->raws[pixel_index];
    }
    XYZ slope_at(iXY coord) {
        return process_info->camera->slope_at(coord.X, coord.Y);
    }



};

 class SceneManager {
public:
    Scene* scene;
    RayEngine RE;
    GUIHandler GUI;
    vector<vector<XYZ*>> raw_output;

    int thread_count = 12;
    vector<RenderThread*> threads;

    int current_resolution_x = 0;
    int current_resolution_y = 0;
    int subdivision_count = 0;
    int current_samples_per_pixel = 0;
    const int block_size = 8;

    int y_increment;
    int x_increment;

    vector<RenderThread::block*> blocks;
    vector<RenderThread::block*> job_stack;

    chrono::steady_clock::time_point render_start;

    SceneManager(Scene* _scene) : scene(_scene) {}

    void init(int resolution_x, int resolution_y) {
        current_resolution_x = resolution_x;
        current_resolution_y = resolution_y;

        

        current_samples_per_pixel = subdivision_count * subdivision_count;

        for (int y = 0; y < current_resolution_y; y++) {
            vector<XYZ*> row;
            for (int x = 0; x < current_resolution_x; x++) {
                row.push_back(new XYZ(0, 0, 0));
            }
            raw_output.push_back(row);
        }

        scene->camera = scene->cameras[0];
        scene->camera->prep(resolution_x,resolution_y,1);
        GUI = GUIHandler(current_resolution_x, current_resolution_y, block_size);
        GUI.create_window();


    }

    void render( int _subdivision_count = 1, int mode = 0) {
        subdivision_count = _subdivision_count;

        y_increment = ceil(current_resolution_y / block_size);
        x_increment = ceil(current_resolution_x / block_size);

        render_start = chrono::high_resolution_clock::now();
        prep();
        if(threads.size()==0) spawn_threads();
        update_threads();
        //enqueue_rays();
        if (mode == 0) prepGUI();
        run_threaded_engine(mode);
        //_run_engine(true);
        delete RE.data->flat_bvh;
        delete RE.data;

    }
    void hold_window() {
        GUI.hold_window();
    }

private:
    void prep() {
        auto prep_start = chrono::high_resolution_clock::now();

        scene->prep(current_resolution_x, current_resolution_y, subdivision_count);
        RE.load_scene(scene->package());

        auto prep_end = chrono::high_resolution_clock::now();
    }
    void update_threads() {
        for (int i = 0; i < thread_count; i++) {
            RenderThread::processing_info* config = create_thread_config();
            delete threads[i]->process_info;
            threads[i]->process_info = config;
            threads[i]->idle = true;
            threads[i]->halt_ops = false;
            threads[i]->stop_thread = false;

        }
    }
    void spawn_threads() {
        for (int i = 0; i < thread_count; i++) {
            RenderThread::processing_info* config = create_thread_config();
            RenderThread* thread = new RenderThread(config);
            threads.push_back(thread);
            SetThreadPriority(thread->worker.native_handle(), 0);
        }
        for (int block_index_y = 0; block_index_y < y_increment; block_index_y++) {
            for (int block_index_x = 0; block_index_x < x_increment; block_index_x++) {
                job_stack.push_back(new RenderThread::block(block_size, block_index_x, block_index_y));
                blocks.push_back(job_stack.back());
            }
        }
        assign_iterative_groups(job_stack);
    }
    struct render_stats {
        time_point start_time;
        time_point now;
        time_point last_refresh = chrono::high_resolution_clock::now();
        long long parent_rays = 0;
        long long child_rays = 0;
        long long parent_rays_last = 0;
        long long child_rays_last = 0;
        double percent_last = 0;
        long long pixels_done = 0;
        double percent_done = 0;
        vector<long long> parent_rays_rate_history;
        vector<long long> child_rays_rate_history;
        vector<double>    percent_rate_history;
        int remaining_threads;
        render_stats(int history_size_rays, int history_size_rate) {
            parent_rays_rate_history = vector<long long>(history_size_rays, 0);
            child_rays_rate_history = vector<long long>(history_size_rays, 0);
            percent_rate_history = vector<double>(history_size_rate, 0);
        }
    };
    struct pixel_strip {
        int strip_start;
        iXY block;
        vector<XYZ> outputs;
        pixel_strip(int start, iXY _block) {
            strip_start = start;
            block = _block;
        }
    };
    void assign_iterative_groups(vector<RenderThread::block*>& job_stack) {
        double jobs_per = (double)job_stack.size() / threads.size();
        double amount_given = 0;
        for (int i = 0; i < thread_count; i++) {
            RenderThread* render_thread = threads[i];
            render_thread->mode = 1;
            double num_now = amount_given + jobs_per;
            double to_do = num_now - ceil(amount_given);
            amount_given = num_now;
            if (i == thread_count - 1) num_now = ceil(num_now);
            for (int j = 0; j < to_do; j++) {
                if(job_stack.size()==0) break;
                int job_index = rand() % job_stack.size();
                RenderThread::block* job = job_stack[job_index];
                render_thread->blocks.push_back(job);
                job_stack.erase(job_stack.begin() + job_index);
            }
        }
    }
    void start_threads() {
        for (int i = 0; i < thread_count; i++) {
            RenderThread* render_thread = threads[i];
            render_thread->halt_ops = false;
            render_thread->idle = false;
        }
    }
    void halt_threads() {
        for (int i = 0; i < thread_count; i++) {
            RenderThread* render_thread = threads[i];
            render_thread->halt_ops = true;
        }
    }
    void unhalt_threads() {
        for (int i = 0; i < thread_count; i++) {
            RenderThread* render_thread = threads[i];
            render_thread->halt_ops = false;
        }
    }
    void reset_blocks(vector<RenderThread::block*>& blocks) {
        for (int i = 0;i < blocks.size(); i++) {
            RenderThread::block* block = blocks[i];
            block->iterations_done = 0;
            for (int j = 0; j < block->raws.size(); j++) {
                *block->raws[j] = XYZ();
            }
        }
    }
    void gather_thread_stats(render_stats& rStat) {
        rStat.parent_rays = 0;
        rStat.child_rays = 0;
        rStat.pixels_done = 0;
        for (int i = 0; i < thread_count; i++) {
            RenderThread* render_thread = threads[i];
            rStat.parent_rays += render_thread->process_info->parent_rays_cast;
            rStat.child_rays += render_thread->process_info->child_rays_cast;
            rStat.pixels_done += render_thread->process_info->pixels_done;
        }
    }
    void manage_threads(render_stats& rStat, vector<RenderThread::block*>& job_stack) {
        time_point now = chrono::high_resolution_clock::now();
        for (int i = 0; i < thread_count; i++) {
            RenderThread* render_thread = threads[i];
            bool is_idle = render_thread->idle.load(std::memory_order_relaxed);
            bool is_stopped = render_thread->stop_thread.load(std::memory_order_relaxed);
            if (is_idle && !is_stopped) {
                if (job_stack.size() > 0) {
                    RenderThread::block* job_block = job_stack.back();
                    GUI.set_focus_position(i, job_block->offset.X, job_block->offset.Y);
                    render_thread->blocks.push_back(job_block);
                    job_stack.pop_back();
                    render_thread->idle = false;
                }
                else {
                    render_thread->stop_thread = true;
                    rStat.remaining_threads--;
                    GUI.set_focus_position(i, -10000, -10000);
                }
            }
        }
    }
    iXY get_pixel_offset(int index) {
        return iXY(index % block_size, block_size - index / block_size - 1);
    }
    
    void get_screen_update(vector<RenderThread::block*>& blocks) {
        for (int i = 0;i < blocks.size(); i++) {
            RenderThread::block* block = blocks[i];
            iXY block_offset = block->offset;
            for (int pixel_index = 0; pixel_index < block_size * block_size; pixel_index++) {
                iXY internal_offset = get_pixel_offset(pixel_index);
                iXY final_coord = block_offset + internal_offset;
                (*raw_output[final_coord.Y][final_coord.X]) = *block->raws[pixel_index];
                XYZ color = (*raw_output[final_coord.Y][final_coord.X], 1, 1);
                GUI.commit_pixel(color, final_coord.X, final_coord.Y);
            }
        }
    }
    void refresh_display() {
        GUI.commit_canvas();
        GUI.handle_events();
    }
    void run_threaded_engine(int mode = 0) {
        render_stats rStat(120, 300);
        rStat.remaining_threads = thread_count;

        
        switch (mode) {
        case 0:
            break;
        case 1:
            start_threads();
            while (true) {
                rStat.now = chrono::high_resolution_clock::now();
                if (chrono::duration_cast<chrono::milliseconds>(rStat.now - rStat.last_refresh).count() > 16) {
                    for (int i = 0; i < threads.size(); i++) {
                        auto& thread = threads[i];
                        if (!thread->halt_ops.load(std::memory_order_relaxed)
                            && thread->idle.load(std::memory_order_relaxed)
                            ) {
                            thread->halt_ops.store(true,std::memory_order_relaxed);
                            rStat.remaining_threads--;
                        }
                    }
                }
                if (rStat.remaining_threads <= 0) {
                    break;
                }

            }
            get_screen_update(blocks);
            GUI.handle_events();
            break;
        }
        for (int i = 0; i < blocks.size(); i++) {
            RenderThread::block* block = blocks[i];
            iXY block_offset = block->offset;
            for (int pixel_index = 0; pixel_index < block->raws.size(); pixel_index++) {
                iXY internal_offset = iXY(pixel_index % block_size, block_size - pixel_index / block_size - 1);
                iXY final_position = block_offset + internal_offset;
                XYZ color = ImageHandler::post_process_pixel(*block->raws[pixel_index], 1, 1);
                GUI.commit_pixel(color, final_position.X, final_position.Y);
            }
        }
        refresh_canvas();
        cout << endl;

        auto end_time = chrono::high_resolution_clock::now();
        double millis = chrono::duration_cast<chrono::milliseconds>(end_time - render_start).count();
        cout << "Ray Casting Done [" << to_string(millis) << "ms]" << endl;
        //cout << "Approximate speed: " << intToEng((rays_cast / (millis / 1000.0))) << " rays/s";
    }
    void prepGUI() {
        for (int i = 0; i < thread_count; i++) {
            GUI.add_focus();
        }
    }
    RenderThread::processing_info* create_thread_config() {
        RenderThread::processing_info* process_info = new RenderThread::processing_info();

        int max_bounces = 2;

        process_info->emit_coord = scene->camera->position;
        process_info->starting_coefficient = XYZ(1, 1, 1) / current_samples_per_pixel;

        process_info->res_y = current_resolution_y;
        process_info->res_x = current_resolution_x;
        process_info->y_increment = y_increment;
        process_info->x_increment = x_increment;
        process_info->block_size = block_size;
        process_info->samples_per_pixel = current_samples_per_pixel;
        process_info->pixels_per_block = block_size * block_size;

        process_info->camera = scene->camera;//camera->clone();
        process_info->RE = new RayEngine(RE);

        return process_info;
    }


    void refresh_canvas() {
        for (int y = 0; y < current_resolution_y; y++) {
            for (int x = 0; x < current_resolution_x; x++) {
                XYZ color = ImageHandler::post_process_pixel(*raw_output[y][x], 1, 1);
                GUI.commit_pixel(color, x, y);
            }
        }
        GUI.commit_canvas();
    }
};



