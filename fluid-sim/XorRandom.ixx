
#include <math.h>
#include<random>
#include <stdint.h>
export module XorRandom;

using namespace std;

/*Xorshiro256+ pseudorandom start. https://prng.di.unimi.it/*/

export class XorGen {
public:

    void seed(uint64_t n1, uint64_t n2, uint64_t n3, uint64_t n4) {
        xe_seed[0] = n1;
        xe_seed[1] = n2;
        xe_seed[2] = n3;
        xe_seed[3] = n4;
    }
    uint64_t xe_next(void) {
        const uint64_t result = xe_seed[0] + xe_seed[3];

        const uint64_t t = xe_seed[1] << 17;

        xe_seed[2] ^= xe_seed[0];
        xe_seed[3] ^= xe_seed[1];
        xe_seed[1] ^= xe_seed[2];
        xe_seed[0] ^= xe_seed[3];

        xe_seed[2] ^= t;

        xe_seed[3] = rotl(xe_seed[3], 45);

        return result;
    }
    void xe_jump(void) {
        const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        uint64_t s2 = 0;
        uint64_t s3 = 0;
        for (int i = 0; i < sizeof JUMP / sizeof * JUMP; i++)
            for (int b = 0; b < 64; b++) {
                if (JUMP[i] & UINT64_C(1) << b) {
                    s0 ^= xe_seed[0];
                    s1 ^= xe_seed[1];
                    s2 ^= xe_seed[2];
                    s3 ^= xe_seed[3];
                }
                xe_next();
            }

        xe_seed[0] = s0;
        xe_seed[1] = s1;
        xe_seed[2] = s2;
        xe_seed[3] = s3;
    }

    void xe_long_jump(void) {
        const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        uint64_t s2 = 0;
        uint64_t s3 = 0;
        for (int i = 0; i < sizeof LONG_JUMP / sizeof * LONG_JUMP; i++)
            for (int b = 0; b < 64; b++) {
                if (LONG_JUMP[i] & UINT64_C(1) << b) {
                    s0 ^= xe_seed[0];
                    s1 ^= xe_seed[1];
                    s2 ^= xe_seed[2];
                    s3 ^= xe_seed[3];
                }
                xe_next();
            }

        xe_seed[0] = s0;
        xe_seed[1] = s1;
        xe_seed[2] = s2;
        xe_seed[3] = s3;
    }
    /*Xorshiro256+ pseudorandom end*/

    float xe_frand() {
        return (xe_next() >> 11) * 0x1.0p-53;
    }

    float rand_frand() //https://stackoverflow.com/a/2704552
    {
        return (float)rand() / RAND_MAX;
    }

    float fRand(float fMin, float fMax) //https://stackoverflow.com/a/2704552
    {
        float f = xe_frand();
        return fMin + f * (fMax - fMin);
    }
private:
    uint64_t xe_seed[4] = { 0,0,0,0 };

    inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

