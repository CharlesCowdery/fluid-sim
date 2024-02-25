
#include <string>
#include <vector>;
#include <math.h>;
import XYZ;
import Matrix;
import VecLib;
import XorRandom;
export module Materials;

#define PI 3.14159265358979323846264

using namespace std;

export class XYZTexture {
public:
    int resolution_x;
    int resolution_y;
    double U_scalar;
    double V_scalar;
    vector<vector<XYZ>> data;
    //XYZTexture(vector<vector<XYZ>>* data_ptr) {
    //    data = *data_ptr;
    //}
    //XYZTexture(string file_name) {
    //    const string file_n = file_name;
    //    int width, height, original_no_channels;
    //    int desired_no_channels = 3;
    //    unsigned char* img = stbi_load(file_n.c_str(), &width, &height, &original_no_channels, desired_no_channels);
    //    string output = padString("Attempting to fetch texture at path: " + file_n + "", ".", 100);
    //    cout << output;
    //    if (img == NULL) {
    //        cout << "!Failed!";
    //        throw invalid_argument("");
    //    }
    //    cout << "[Done]" << endl;
    //    string process_string = padString(
    //        "    ->Loading: [" + to_string(width) +
    //        "px : " + to_string(height) +
    //        "px] Channels: [original: " + to_string(original_no_channels) +
    //        " ; loaded: " + to_string(desired_no_channels) + "]"
    //        , ".", 100);
    //    cout << process_string << flush;
    //
    //    resolution_x = width;
    //    resolution_y = height;
    //    for (int y = 0; y < resolution_y; y++) {
    //        data.push_back(vector<XYZ>());
    //        for (int x = 0; x < resolution_x; x++) {
    //            int position = y * resolution_x * 3 + x * 3;
    //            float r = (float)img[position];
    //            float g = (float)img[position + 1];
    //            float b = (float)img[position + 2];
    //            XYZ color = XYZ(r / 256.0, g / 256.0, b / 256.0);
    //            color = (color - XYZ(0.5)) * 2;
    //            data.back().push_back(color);
    //            ;
    //        }
    //    }
    //    delete[] img;
    //    U_scalar = resolution_x;
    //    V_scalar = resolution_y;
    //    cout << "[Done]" << endl;
    //}
    void prep() {
        U_scalar = resolution_x;
        V_scalar = resolution_y;
    }
    XYZ lookup(int x, int y) {
        return data[y][x];
    }
    XYZ getUV(double U, double V) {
        V = 1 - V;
        int pos_x = floor(U_scalar * U);
        int pos_y = floor(V_scalar * V);
        return lookup(pos_x, pos_y);
    }
    XYZ getPixelLinear(double U, double V) {
        double x_1 = U_scalar * U - 0.5 * U_scalar;
        double x_2 = U_scalar * U + 0.5 * U_scalar;
        double y_1 = V_scalar * V - 0.5 * V_scalar;
        double y_2 = V_scalar * V + 0.5 * V_scalar;
        int pos_x_1 = std::max(0, std::min(resolution_x - 1, (int)floor(x_1)));
        int pos_x_2 = std::max(0, std::min(resolution_x - 1, (int)floor(x_2)));
        int pos_y_1 = std::max(0, std::min(resolution_y - 1, (int)floor(y_1)));
        int pos_y_2 = std::max(0, std::min(resolution_y - 1, (int)floor(y_2)));
        XYZ c1 = lookup(pos_x_1, pos_y_1);
        XYZ c2 = lookup(pos_x_2, pos_y_1);
        XYZ c3 = lookup(pos_x_1, pos_y_2);
        XYZ c4 = lookup(pos_x_2, pos_y_2);
        double f1 = pos_x_2 - x_1;
        double f2 = x_2 - pos_x_2;
        double f3 = pos_y_2 - y_1;
        double f4 = y_2 - pos_y_2;
        XYZ c_m1 = f1 * c1 + f2 * c2;
        XYZ c_m2 = f1 * c3 + f2 * c4;
        return f3 * c_m1 + f4 * c_m2;
    }
};

export class Parameter {
public:
    XYZ static_value = 0;
    XYZTexture* texture = nullptr;
    Parameter(float X, float Y, float Z) : static_value(XYZ(X, Y, Z)) {}
    Parameter(XYZ value) : static_value(value) {}
    Parameter(XYZTexture* texture_ptr) : texture(texture_ptr) {}
    //Parameter(string file_name) : texture(new XYZTexture(file_name)) {}
    void set_static(double value) {
        static_value.X = value;
    }
    void set_static(XYZ value) {
        static_value = XYZ(value);
    }
    void prep() {
        if (texture != nullptr) {
            texture->prep();
        }
    }
    void set_texture(XYZTexture* texture_ptr, bool free = false) {
        if (texture != nullptr) {
            if (free) {
                delete texture;
            }
        }
        texture = texture_ptr;
    }
    void set_texture(string file_name, bool free = false) {
        //set_texture(new XYZTexture(file_name), free);
    }
    XYZ getXYZ(double U, double V) {
        if (texture != nullptr) {
            return texture->getUV(U, V);
        }
        else {
            return static_value;
        }
    }
    double getSingle(double U, double V) {
        if (texture != nullptr) {
            return texture->getUV(U, V).X;
        }
        else {
            return static_value.X;
        }
    }
};

export class MaterialSample {
public:
    XYZ color;
    float roughness;
    float metallic;
    float specular;
    XYZ emissive;
    XYZ normal;

    float k = 0;
    float a_2 = 0;
    float spec_f = 0;
    float diff_f = 0;
    float diff_spread = 0;
    XYZ diff_c = XYZ();
    XYZ diff_t = XYZ();
    XYZ spec_color = XYZ();
    XYZ I_spec = XYZ();

    MaterialSample(XYZ _color, float _roughness, float _metallic, float _specular, XYZ _emissive, XYZ _normal) {
        color = _color;
        roughness = _roughness;
        metallic = _metallic;
        specular = _specular;
        emissive = _emissive;
        normal = _normal;

        k = pow(roughness + 1, 2) / 8;
        a_2 = roughness * roughness;
        spec_color = get_specular_color();
        I_spec = 1 - get_fresnel_0();
        spec_f = get_specular_factor();
        diff_f = get_diffuse_factor();
        diff_c = get_diffuse_color();
        diff_t = diff_c * diff_f;
        float r_f = roughness - 0.2;
        diff_spread = (r_f * r_f) * PI / 2;
    }
    float get_diffuse_factor() const {
        return 1 - get_specular_factor();
    }
    XYZ get_fresnel_0() const {
        return XYZ::linear_mix(metallic, XYZ(0.04), color);
    }
    XYZ get_diffuse_reflectance() const {
        return color * (1 - metallic);
    }
    float get_specular_factor() const {
        return std::min(std::max(metallic, specular), (float)1.0);
    }
    XYZ get_specular_factor_v2(float dot_NI) const {
        return fast_fresnel(dot_NI);
    }
    XYZ get_diffuse_factor_v2(float dot_NI) const {
        return XYZ(1) - fast_fresnel(dot_NI);
    }
    XYZ get_specular_color() const {
        return XYZ::linear_mix(metallic, specular * XYZ(1, 1, 1), color);
    }
    XYZ get_diffuse_color() const {
        return get_diffuse_reflectance() / PI;
    }
    XYZ get_fresnel(XYZ light_slope, XYZ normal) const {
        XYZ specular_color = get_fresnel_0();
        auto second_term = (1 - specular_color) * pow(1 - XYZ::dot(light_slope, normal), 5);
        return specular_color + second_term;
    }
    XYZ fast_fresnel(float dot_NI) const {
        float g = 1 - dot_NI;
        return I_spec * (g * g * g * g * g);
    }
    float get_normal_distribution_beta(const XYZ& normal, const XYZ& half_vector) const {
        float a = roughness;
        float dot = XYZ::dot(normal, half_vector);
        float exponent = -(1 - pow(dot, 2)) / (pow(a * dot, 2));
        float base = 1 / (PI * pow(a, 2) * pow(dot, 4));
        float final = base * exp(exponent);

        return final;
    }
    float get_normal_distribution_GGXTR(const XYZ& normal, const XYZ& half_vector) const {
        float a = roughness;
        float dot = XYZ::dot(normal, half_vector);
        float final = (a * a)
            /
            (PI * pow((dot * dot) * (a * a - 1) + 1, 2));

        return final;
    }
    float fast_normal_dist(const float dot_NH) const {
        float a_2 = roughness * roughness;
        float g = dot_NH * dot_NH * (a_2 - 1) + 1;
        return dot_NH * (a_2) / (PI * g * g);
    }
    float geoSchlickGGX(const XYZ& normal, const XYZ& vector, float k) const {

        float dot = XYZ::dot(normal, vector);

        return dot / (dot * (1 - k) + k);
        //return 1;

    }
    float fastGeo_both(const float dot_NO, const float dot_NI) const {
        return dot_NO / (dot_NO * (1 - k) + k) * dot_NI / (dot_NI * (1 - k) + k);
    }
    XYZ diffuse_BRDF() const {

        //return get_diffuse_factor_v2(dot_NI) * get_diffuse_reflectance();
        return get_diffuse_reflectance();
    }
    XYZ diffuse_BRDF(float dot_NI) const {

        //return get_diffuse_factor_v2(dot_NI) * get_diffuse_reflectance();
        return diffuse_BRDF();
    }
    XYZ diffuse_BRDF_unweighted(float dot_NI) const {
        return get_diffuse_factor_v2(dot_NI) * get_diffuse_reflectance();
    }
    XYZ specular_BRDF(const XYZ& normal, const XYZ& input_slope, XYZ& output_slope) const {
        float dot_NI = XYZ::dot(normal, input_slope);
        float dot_NO = XYZ::dot(normal, output_slope);
        if (dot_NI <= 0 || dot_NO <= 0) {
            return XYZ(0, 0, 0);
        }
        XYZ half_vector = XYZ::normalize(XYZ::add(input_slope, output_slope));
        float dot_HO = XYZ::dot(half_vector, output_slope);
        float dot_NH = XYZ::dot(normal, half_vector);
        XYZ fresnel = fast_fresnel(dot_HO);
        float geo = fastGeo_both(dot_NO, dot_NI);
        float normal_dist = fast_normal_dist(dot_NH);
        float divisor = 4 * dot_NO;

        return geo * normal_dist * fresnel;// / divisor;
    }
    XYZ fast_BRDF_co(const XYZ& normal, const XYZ& input_slope, XYZ& output_slope) const {
        float dot_NI = XYZ::dot(normal, input_slope);
        float dot_NO = XYZ::dot(normal, output_slope);
        if (dot_NI <= 0 || dot_NO <= 0) {
            return XYZ(0, 0, 0);
        }
        XYZ half_vector = XYZ::normalize(XYZ::add(input_slope, output_slope));
        float dot_HO = XYZ::dot(half_vector, output_slope);
        float dot_NH = XYZ::dot(normal, half_vector);
        XYZ fresnel = fast_fresnel(dot_HO);
        float geo = fastGeo_both(dot_NO, dot_NI);
        float normal_dist = fast_normal_dist(dot_NH);
        float divisor = 4 * dot_NI * dot_NO;

        XYZ specular_return = (geo * normal_dist * fresnel) / divisor;

        XYZ light_remainer = XYZ(1) - specular_return;

        return (specular_return + light_remainer*diffuse_BRDF()) * dot_NI;

    }
    XYZ random_bounce(XorGen& G, const Matrix3x3& diffuse_rotation, const Matrix3x3& reflection_rotation) const {
        float prob = G.fRand(0, 1);
        if (prob < diff_f) {
            return Matrix3x3::aligned_random(G, PI / 2, diffuse_rotation);
        }
        else {
            return Matrix3x3::aligned_random(G, diff_spread, reflection_rotation);
        }
    }
    XYZ biased_diffuse_bounce(XorGen& G, const Matrix3x3& diffuse_rotation, float r1_min = 0, float r1_max = 1, float r2_min = 0, float r2_max = 1) const {
        return Matrix3x3::aligned_biased_hemi(G, diffuse_rotation, r1_min, r1_max, r2_min, r2_max);
    }
    XYZ unbiased_diffuse_bounce(XorGen& G, const Matrix3x3& diffuse_rotation) const {
        return Matrix3x3::aligned_unbiased_hemi(G, diffuse_rotation);
    }
    XYZ reflective_bounce(XorGen& G, const Matrix3x3& reflection_rotation) const {
        return Matrix3x3::aligned_random(G, diff_spread, reflection_rotation);
    }

    XYZ calculate_emissions() const {
        return emissive;
    }
};

export class Material {
public:
    Parameter color = Parameter(0);
    Parameter roughness = Parameter(0.1);
    Parameter metallic = Parameter(0);
    Parameter specular = Parameter(0);
    Parameter emissive = Parameter(0);
    Parameter normal = Parameter(0);

    bool use_normals = false;

    Material() {}

    void prep() {
        color.prep();
        roughness.prep();
        metallic.prep();
        specular.prep();
        emissive.prep();
        normal.prep();
    }
    MaterialSample sample_UV(float U, float V) {
        return MaterialSample(
            color.getXYZ(U, V),
            roughness.getSingle(U, V),
            metallic.getSingle(U, V),
            specular.getSingle(U, V),
            emissive.getXYZ(U, V),
            normal.getXYZ(U, V)
        );
    }
};