
import XYZ;
import XorRandom;
#include <vector>;
#include <immintrin.h>
export module VecLib;

#define PI 3.14159265358979323846264

#define LOOKUP_SIZE_TRIG 1024
#define LOOKUP_TRIG_MASK LOOKUP_SIZE_TRIG-1
#define LOOKUP_SIZE_BIASED_HEMI 1024*32
#define LOOKUP_BIASED_HEMI_MASK LOOKUP_SIZE_BIASED_HEMI-1

export float arccos_lookup[LOOKUP_SIZE_TRIG];
export float sin_lookup[LOOKUP_SIZE_TRIG];
export float cos_lookup[LOOKUP_SIZE_TRIG];
export XYZ   biased_hemi_lookup[LOOKUP_SIZE_BIASED_HEMI];

export namespace VecLib {
    
    __m256 sgn_fast(__m256& x) //credit to Peter Cordes https://stackoverflow.com/a/41353450
    {
        __m256 negzero = _mm256_set1_ps(-0.0f);

        // using _mm_setzero_ps() here might actually be better without AVX, since xor-zeroing is as cheap as a copy but starts a new dependency chain
        //__m128 nonzero = _mm_cmpneq_ps(x, negzero);  // -0.0 == 0.0 in IEEE floating point

        __m256 x_signbit = _mm256_and_ps(x, negzero);
        return _mm256_or_ps(_mm256_set1_ps(1.0f), x_signbit);
    }
    void cross_avx(const m256_vec3& v1, const m256_vec3& v2, m256_vec3& output) {
        __m256 c1 = _mm256_mul_ps(v1.Z, v2.Y);
        __m256 c2 = _mm256_mul_ps(v1.X, v2.Z);
        __m256 c3 = _mm256_mul_ps(v1.Y, v2.X);
        output.X = _mm256_fmsub_ps(v1.Y, v2.Z, c1);
        output.Y = _mm256_fmsub_ps(v1.Z, v2.X, c2);
        output.Z = _mm256_fmsub_ps(v1.X, v2.Y, c3);
    }
    void dot_avx(const m256_vec3& v1, const m256_vec3& v2, __m256& output) {
        output = _mm256_fmadd_ps(
            v1.Z,
            v2.Z,
            _mm256_fmadd_ps(
                v1.Y,
                v2.Y,
                _mm256_mul_ps(
                    v1.X,
                    v2.X
                )
            )
        );
    }
    __m256 inline_dot_avx(const m256_vec3& v1, const m256_vec3& v2) {
        __m256 output;
        dot_avx(v1, v2, output);
        return output;
    }
    void dot_mul_avx(const m256_vec3& v1, const m256_vec3& v2, const __m256& v3, __m256& output) {
        output = _mm256_mul_ps(
            v3,
            _mm256_fmadd_ps(
                v1.Z,
                v2.Z,
                _mm256_fmadd_ps(
                    v1.Y,
                    v2.Y,
                    _mm256_mul_ps(
                        v1.X,
                        v2.X
                    )
                )
            )
        );
    }
    double poly_acos(float x) {
        return (-0.69813170079773212 * x * x - 0.87266462599716477) * x + 1.5707963267948966;
    }
     float Lsin(float input) {
        if (input < 0) input = PI - input;
        int index = ((int)(input * (LOOKUP_SIZE_TRIG / (PI * 2)))) & LOOKUP_TRIG_MASK;
        return sin_lookup[index];
    }
     float Lcos(float input) {
        if (input < 0) input = 2 * PI - input;
        int index = ((int)(input * (LOOKUP_SIZE_TRIG / (PI * 2)))) & LOOKUP_TRIG_MASK;
        return cos_lookup[index];
    }
     float Lacos(float input) {
        int index = ((int)((input + 1) / 2.0 * LOOKUP_SIZE_TRIG)) & LOOKUP_TRIG_MASK;
        return arccos_lookup[index];
    }
     XYZ gen_random_point_on_sphere(XorGen& G) {
        double r1 = G.fRand(0, 1);
        double r2 = G.fRand(0, 1);
        double f1 = Lacos(2 * r1 - 1) - PI / 2;
        double f2 = r2 * 2 * PI;
        return XYZ(Lcos(f1) * Lcos(f2), Lsin(f1), Lcos(f1) * Lsin(f2));
    }
    //r1 is the vertical component, r2 is the rotational component
     XYZ generate_biased_random_hemi_v2(XorGen& G ,float r1_min = 0, float r1_max = 1, float r2_min = 0, float r2_max = 1) {//https://www.rorydriscoll.com/2009/01/07/better-sampling/
        const float r1 = G.fRand(r1_min, r1_max);
        const float r2 = G.fRand(r2_min, r2_max);
        const float r = sqrt(r1);
        const float theta = 2 * PI * r2;

        const float x = r * Lcos(theta);
        const float y = r * Lsin(theta);

        return XYZ(x, sqrt(1 - r1), y);
    }
     XYZ lookup_biased_random_hemi(XorGen& G) {
        int index = G.fRand(0, 1) * LOOKUP_BIASED_HEMI_MASK;
        return biased_hemi_lookup[index];
    }
     XYZ generate_biased_random_hemi(XorGen& G) {
        XYZ pos = XYZ(0, 1, 0) + gen_random_point_on_sphere(G);
        return XYZ::slope(XYZ(0), pos);
    }
     void prep(XorGen& G) {
        for (int i = 0; i < LOOKUP_SIZE_TRIG; i++) {
            double pi2_scalar = ((double)LOOKUP_SIZE_TRIG) / 2.0 / PI;
            double scalar2 = (double)LOOKUP_SIZE_TRIG / 2;
            sin_lookup[i] = sin((double)i / pi2_scalar);
            cos_lookup[i] = cos((double)i / pi2_scalar);
            arccos_lookup[i] = acos((double)i / scalar2 - 1);
        }
        for (int i = 0; i < LOOKUP_SIZE_BIASED_HEMI; i++) {
            biased_hemi_lookup[i] = generate_biased_random_hemi(G);
        }
    }
     XYZ generate_unbiased_random_hemi(XorGen& G) {
        XYZ pos = gen_random_point_on_sphere(G);
        if (pos.Y < 0) pos.Y *= -1;
        return XYZ::slope(0, pos);
    }
     XYZ biased_random_hemi(XorGen& G, float r1_min = 0, float r1_max = 1, float r2_min = 0, float r2_max = 1) {
        return generate_biased_random_hemi_v2(G, r1_min, r1_max, r2_min, r2_max);
    }
     XYZ unbiased_random_hemi(XorGen& G) {
        return generate_unbiased_random_hemi(G);
    }
     XYZ lookup_random_cone(XorGen& G, float spread) {
        float y_f = G.fRand(0, spread);
        float z_f = G.fRand(0, 2 * PI);
        return XYZ(Lsin(y_f) * Lcos(z_f), Lcos(y_f), Lsin(y_f) * Lsin(z_f));
    }
     XYZ y_random_cone(XorGen& G,float spread) {
        float y = G.fRand(1, spread);
        float rot = G.fRand(0, PI);
        float f_y = sqrt(1 - y * y);
        return XYZ(f_y * cos(rot), y, f_y * sin(rot));
    }
     XYZ aligned_random(XorGen& G, float spread, const Quat& r) {
        return Quat::applyRotation(lookup_random_cone(G,spread), r);
    }
     float surface_area(const XYZ& max, const XYZ& min) {
        float l_x = max.X - min.X;
        float l_y = max.Y - min.Y;
        float l_z = max.Z - min.Z;
        return (l_x * l_y + l_x * l_z + l_y * l_z) * 2;
    }
     float volume(const XYZ& max, const XYZ& min) {
        float l_x = max.X - min.X;
        float l_y = max.Y - min.Y;
        float l_z = max.Z - min.Z;
        return (l_x * l_y * l_z);
    }
     bool between(const XYZ& p1, const XYZ& p2, const XYZ test) {
        return (
            (p1.X < test.X && test.X < p1.X) &&
            (p1.Y < test.Y && test.Y < p1.Y) &&
            (p1.Z < test.Z && test.Z < p1.Z)
            );
    }
     bool betweenX(const XYZ& p1, const XYZ& p2, const XYZ test) {
        return (
            (p1.X < test.X && test.X < p1.X)
            );
    }
     bool betweenY(const XYZ& p1, const XYZ& p2, const XYZ test) {
        return (
            (p1.Y < test.Y && test.Y < p1.Y)
            );
    }
     bool betweenZ(const XYZ& p1, const XYZ& p2, const XYZ test) {
        return (
            (p1.Z < test.Z && test.Z < p1.Z)
            );
    }
     bool volumeClip(const XYZ& max_1, const XYZ& min_1, const XYZ& max_2, const XYZ& min_2) {
        bool term1 = (max_1.X >= min_2.X && max_2.X >= min_1.X);
        bool term2 = (max_1.Y >= min_2.Y && max_2.Y >= min_1.Y);
        bool term3 = (max_1.Z >= min_2.Z && max_2.Z >= min_1.Z);
        return term1 && term2 && term3;
        /*bool term1 = (
            (((max_2.X < max_1.X) && (max_2.X > min_1.X)) || ((min_2.X < max_1.X) && (min_2.X > min_1.X))) &&
            (((max_2.Y < max_1.Y) && (max_2.Y > min_1.Y)) || ((min_2.Y < max_1.Y) && (min_2.Y > min_1.Y))) &&
            (((max_2.Z < max_1.Z) && (max_2.Z > min_1.Z)) || ((min_2.Z < max_1.Z) && (min_2.Z > min_1.Z)))
            );
        return term1;*/
    }
     bool volumeContains(const XYZ& max, const XYZ& min, const XYZ& point) {
        bool t1 = (point.X<max.X && point.X>min.X);
        bool t2 = (point.Y<max.Y && point.Y>min.Y);
        bool t3 = (point.Z<max.Z && point.Z>min.Z);
        return t1 && t2 && t3;
    }
}