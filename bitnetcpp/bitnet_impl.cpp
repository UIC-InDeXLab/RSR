#include <cstring>
#include <random>
#include "globals.hpp"

void quantize_i2(const float * src, void * dst) {
    // 2 bits per weight
    int n = K;
    double i2_scale = 1;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, n * sizeof(uint8_t));

    // q8 -> 0, 1, 2
    //       |  |  |
    //      -1, 0, 1

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i=0; i<n; i++) {
        int group_idx = i / 4;
        int group_pos = i % 4;
        uint8_t temp = (q8[i] << (6 - 2 * group_pos));
        i2_weight[group_idx] |= temp;
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;
}

void ggml_vec_dot_i2_i8(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t *    x = (uint8_t*)vx;
    const int8_t  *    y = (int8_t*)vy;

    int sumi = 0;

    for (int i = 0; i < n / 4; ++i) {
        uint8_t xi = x[i];

        for (int k = 0; k < 4; ++k) {
            uint8_t two_bits = (xi >> (2 * k)) & 0x03;
            int8_t ternary = (int8_t)two_bits - 1;       // Map {0,1,2} → {-1,0,1}
            float product = ternary * y[i * 4 + (3 - k)];      // Multiply
            sumi += product;
        }
    }
    *s = (float)sumi;
}