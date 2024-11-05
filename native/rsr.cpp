#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "utility"
#include "utils.h"
#include "naive.h"

using namespace std;

// Return <permutation, segmentation>
pair<vector<int>, vector<int>> handle_block(vector<vector<int>> mat_block) {
    int n = mat_block.size();
    int k = mat_block[0].size();

    // Permutation
    vector<int> permutation(n);
    for (int i = 0; i < n; i++) {
        permutation[i] = i;
    }

    sort(permutation.begin(), permutation.end(), [&](int i, int j) {
        // return vec[i] < vec[j]
        return binaryVectorToInt(mat_block[i]) < binaryVectorToInt(mat_block[j]);
    });

    
    // Segmentation
    vector<int> seg(pow(2, k), -1);
    seg[0] = 0;
    for (int row = 0; row < n; row++) {
        int value = binaryVectorToInt(mat_block[permutation[row]]);
        if (seg[value] == -1) {
            seg[value] = row;
        }
    }
    if (seg[seg.size() - 1] == -1) {
        seg[seg.size() - 1] = n;
    }
    int last_one = seg[seg.size() - 1];
    for (int i = seg.size() - 2; i >= 0; i--) {
        if (seg[i] == -1) {
            seg[i] = last_one;
        }
        last_one = seg[i];
    }

    return make_pair(permutation, seg);
}

pair<vector<vector<int>>, vector<vector<int>>> preprocess(vector<vector<int>>& mat, int k) {
    int n = mat.size();

    // Padding
    int padding = (k - n % k) % k;
    for (auto& row : mat) {
        row.resize(row.size() + padding, 0);
    }

    for (int i = 0; i < padding; i++) {
        mat.push_back(vector<int>(n + padding, 0));
    }
    n = n + padding;

    vector<vector<int>> permutations(n / k, vector<int>(n));
    vector<vector<int>> segs(n / k, vector<int>(pow(2, k)));


    // Blocking
    int start;
    int end;
    vector<vector<int>> block(n, vector<int>(k));
    for (int i = 0; i < n / k; i++) {
        // cout << "block " << i + 1 << " out of " << n / k << " blocks" << endl;
        start = i * k;
        end = start + k;
        for (int col = start; col < end; col++) {
            for (int row = 0; row < n; row++) {
                block[row][col - start] = mat[row][col];
            }
        }
        auto per_seg = handle_block(block);
        permutations[i] = per_seg.first;
        segs[i] = per_seg.second;
    }

    return make_pair(permutations, segs);
}

vector<int> rsr_inference(vector<int> v, const vector<vector<int>>& permutations, const vector<vector<int>>& segments, vector<vector<int>> bin_k, int k) {
    int n = permutations[0].size();

    // segmented sums
    vector<vector<int>> us(permutations.size(), vector<int>(pow(2, k)));

    int start;
    int end;
    vector<int> segment;
    vector<int> permutation;
    for (int i = 0; i < permutations.size(); i++) {
        segment = segments[i];
        permutation = permutations[i];

        // Each block
        for (int j = 0; j < segment.size(); j++) {
            start = segment[j];
            if (j < segment.size() - 1) {
                end = segment[j + 1];
            } else {
                end = n;
            }
            // Segmented sum
            for (int index = start; index < end; index++) {
                us[i][j] += v[permutation[index]];
            }           
        }
    }

    vector<int> result(n);

    // Block product to Bin_k
    // TODO: change from here for RSR++
    vector<int> partial_result;
    for (int i = 0; i < us.size(); i++) {
        partial_result = vectorMatrixMultiply(us[i], bin_k);
        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_result[j];
        }
    }
    return result;
}

// int main() {
//     int n = 1024;
//     int k = static_cast<int>(ceil(log2(n) - log2(log2(n))));

//     vector<vector<int>> mat = generateBinaryRandomMatrix(n);
//     vector<int> v = generateRandomVector(n);

//     auto per_segs = preprocess(mat, k);

//     vector<int> result = rsr_inference(v, per_segs.first, per_segs.second, k);
//     vector<int> gt = vectorMatrixMultiply(v, mat);

//     for(int i = 0; i < n; i++) {
//         if (gt[i] != result[i]) {
//             cout << gt[i] << " " << result[i] << endl;
//         }
//     }

//     return 0;
// }