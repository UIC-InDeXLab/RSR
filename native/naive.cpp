#include <vector>
#include "utils.h"

using namespace std;

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
