#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>
#include <random>

using namespace std;

int binaryVectorToInt(const vector<int>& binaryVec) {
    int result = 0;
    int n = binaryVec.size();
    
    for (int i = 0; i < n; ++i) {
        result = (result << 1) | binaryVec[i];  // Left-shift result and add the next bit
    }
    
    return result;
}

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

vector<vector<int>> generateBinaryRandomMatrix(int n) {
    // Initialize the random number generator
    random_device rd;  // Seed
    mt19937 gen(rd()); // Mersenne Twister engine
    uniform_int_distribution<> dis(0, 1); // Distribution that produces 0 or 1

    // Initialize matrix
    vector<vector<int>> matrix(n, vector<int>(n));

    // Populate the matrix with random 0s and 1s
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = dis(gen); // Generate a random 0 or 1
        }
    }

    return matrix;
}

vector<int> generateRandomVector(int n) {
    // Initialize the random number generator
    random_device rd;  // Seed
    mt19937 gen(rd()); // Mersenne Twister engine
    uniform_int_distribution<> dis(0, 100);

    // Initialize matrix
    vector<int> matrix(n);

    // Populate the matrix with random integers
    for (int i = 0; i < n; ++i) {
        matrix[i] = dis(gen); // Generate random integer in range [minValue, maxValue]
    }

    return matrix;
}