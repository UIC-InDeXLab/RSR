#include <cstdint>
#include <random>
#include <chrono>
#include <vector>
#include "utility"
#include "ggml-quants.h"
#include "globals.hpp"
#include "utils.h"
#include "bitnet_impl.h"
#include "rsr_impl.h"

float random_ternary_bit() {
    int r = rand() % 3;  // 0, 1, 2
    return (float)(r - 1);  // -1.0, 0.0, 1.0
}

float *random_ternary_matrix(int rows, int cols) {
    float *matrix = (float *)malloc(rows * cols * sizeof(float));
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            matrix[row * cols + col] = random_ternary_bit();
        }
    }
    return matrix;
}

pair<vector<vector<int>>, vector<vector<int>>> extract_binary_matrices(vector<vector<int>> matrix) {
    vector<vector<int>> B1(M, vector<int>(K, 0));
    vector<vector<int>> B2(M, vector<int>(K, 0));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            B1[i][j] = (matrix[i][j] == 1) ? 1 : 0;
            B2[i][j] = (matrix[i][j] == -1) ? 1 : 0;
        }
    }

    return make_pair(B1, B2);
}

// Bin_k
vector<vector<int>> generateBinaryMatrix(int k) {
    int rows = pow(2, k);  // 2^k rows
    vector<vector<int>> matrix(rows, vector<int>(k, 0));  // Initialize matrix with 0s

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < k; ++j) {
            // Generate the binary value for each position
            matrix[i][k - j - 1] = (i >> j) & 1;  // Extract the j-th bit from i
        }
    }

    return matrix;
}

// auto pre_start = std::chrono::high_resolution_clock::now();
// preprocessor_t1_int8_m6400_k8640_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);
// auto pre_end   = std::chrono::high_resolution_clock::now();
// std::chrono::duration<double, std::milli> elapsed = pre_end - pre_start;
// preprocess_time += double(elapsed.count());

int main(int argc, char* argv[])
{
    if (argc != 3) {
        printf("Invalid arguments: ./main 1024 1024\nfirst: M (matrix row count), second one: K vector size = matrix col count.\n");
        return 1;
    }
    M = std::atoi(argv[1]);
    K = std::atoi(argv[2]);

    
    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate random vector of size K
    float *vector = (float *)malloc(K * sizeof(float));
    for (int i = 0; i < K; i++)
    {
        vector[i] = rand() % (1000 + 1) / (float)(1000);
    }

    // Quantize vector
    int32_t *act_sums = (int32_t *)malloc(sizeof(int32_t));
    float *act_scales = (float *)malloc(sizeof(float));

    int8_t *quant_vector = (int8_t *)malloc(sizeof(int8_t) * K);

    quantize_row_i8_s(vector, quant_vector, K, act_scales, act_sums);


    // A ternary matrix of size M * K from file
    // float *matrix = load_matrix_bin("original_weight_matrix.bin");
    float *matrix = random_ternary_matrix(K, M);
    if (!matrix)
    {
        printf("Error loading matrix...");
        return 1;
    }

    // Init results placeholders
    float *expected = (float *)malloc(M * sizeof(float));
    for (int i = 0; i < M; i++)
    {
        expected[i] = 0;
    }

    float *bitnet = (float *)malloc(M * sizeof(float));
    for (int i = 0; i < M; i++)
    {
        bitnet[i] = 0;
    }

    // printf("matrix:\n");
    // for (int i = 0; i < M; i++)
    // {
    //     for (int j = 0; j < K; j++)
    //     {
    //         printf("%.2f ", matrix[i * K + j]);
    //     }
    //     printf("\n");
    // }

    // Standard multiplication
    // convert vector to float
    float *float_vector = (float *)malloc(sizeof(float) * K);

    // Convert values
    for (int i = 0; i < K; ++i) {
        float_vector[i] = (float)quant_vector[i];
    }

    // printf("vector:\n");
    // for (int i = 0; i < K; i++){
    //     printf("%d ", quant_vector[i]);
    // }
    // printf("\n");

    auto pre_start_naive = std::chrono::high_resolution_clock::now();
    matrixMultiply(1, matrix, float_vector, expected);
    auto post_start_naive = std::chrono::high_resolution_clock::now();

    printf("expected result:\n");
    for (int i = 0; i < 6; i++)
    {
        printf("%.2f ", expected[i]);
    }
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(post_start_naive - pre_start_naive).count();
    printf("\nTime: %lld ns\n", static_cast<long long>(duration));

    // Packed ternary matrix
    uint8_t *quant_mat = (uint8_t *)malloc(sizeof(uint8_t) * M * K / 4);

    for (int i = 0; i < M; i++)
    {
        quantize_i2(matrix + i * K, quant_mat + i * K / 4);
    }

    auto pre_start_bitnet = std::chrono::high_resolution_clock::now();
    // Bitnet multiplication
    for (int i = 0; i < M; i++)
    {
        ggml_vec_dot_i2_i8(K, bitnet + i, 0, quant_mat + i * K / 4, 0, quant_vector, 0, 0);
    }
    auto post_start_bitnet = std::chrono::high_resolution_clock::now();

    printf("bitnet result:\n");
    for (int i = 0; i < 6; i++)
    {
        printf("%0.2f ", bitnet[i]);
    }
    auto bitnet_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(post_start_bitnet - pre_start_bitnet).count();
    printf("\nTime: %lld ns\n", static_cast<long long>(bitnet_duration));

    // RSR Preprocessing
    std::vector<int> rsr_vector(K, 0);
    std::vector<std::vector<int>> rsr_matrix(M, std::vector<int>(K, 0));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            rsr_matrix[j][i] = matrix[i * K + j];
        }
    }
    for (int i = 0; i < K; i++) {
        rsr_vector[i] = quant_vector[i];
    }
    int k = 4;

    switch (M)
    {
    case 1024:
        k = 4;
        break;
    case 2048:
        k = 5;
        break;
    case 4096:
        k = 5;
        break;
    default:
        break;
    }

    auto bin_k = generateBinaryMatrix(k);
    auto pair = extract_binary_matrices(rsr_matrix);
    auto b1 = preprocess(pair.first, k);

    auto b2 = preprocess(pair.second, k);

    auto pre_start_rsr = std::chrono::high_resolution_clock::now();
    auto r1 = rsr_inference(rsr_vector, b1.first, b1.second, bin_k, k);
    auto r2 = rsr_inference(rsr_vector, b2.first, b2.second, bin_k, k);
    auto post_start_rsr = std::chrono::high_resolution_clock::now();

    printf("rsr result:\n");
    for (int i = 0; i < 6; i++)
    {
        printf("%d ", r1[i] - r2[i]);
    }
    auto rsr_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(post_start_rsr - pre_start_rsr).count();
    printf("\nTime: %lld ns\n", static_cast<long long>(rsr_duration));
    
    printf("\n");
    printf("Done\n");
}
