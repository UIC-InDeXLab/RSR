#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>
#include <random>
#include "globals.hpp"

using namespace std;

void matrixMultiply(int N, const float *A, const float *B, float *C)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i * N + j] = 0.0;
            for (int k = 0; k < K; ++k)
            {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

float *load_matrix_bin(const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        perror("Failed to open file");
        return NULL;
    }

    size_t total = M * K;
    float *matrix = (float *)malloc(total * sizeof(float));
    if (!matrix)
    {
        perror("Failed to allocate memory");
        fclose(f);
        return NULL;
    }

    size_t read_count = fread(matrix, sizeof(float), total, f);
    if (read_count != total)
    {
        fprintf(stderr, "Unexpected file size: expected %zu floats, got %zu\n", total, read_count);
        free(matrix);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return matrix;
}

int binaryVectorToInt(const vector<int>& binaryVec) {
    int result = 0;
    int n = binaryVec.size();
    
    for (int i = 0; i < n; ++i) {
        result = (result << 1) | binaryVec[i];  // Left-shift result and add the next bit
    }
    
    return result;
}

vector<int> vectorMatrixMultiply(const vector<int>& vec, const vector<vector<int>>& mat) {
    int n = vec.size();

    // Initialize the result vector with zeros
    vector<int> result(n, 0);

    // Perform vector-matrix multiplication
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += vec[j] * mat[j][i];
        }
    }

    return result;
}