#include <random>

void quantize_i2(const float * src, void * dst);
void ggml_vec_dot_i2_i8(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);