#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include "utils.h"
#include "naive.h"
#include "rsr.h"

using namespace std;

vector<int> step_three(vector<int> u, int k) {
    vector<int> result(k);
    int sum;
    for (int i = k; i > 0; i--) {
        sum = 0;
        for (int j = 1; j < pow(2, i); j += 2) {
            sum += u[j];
        }
        result[i - 1] = sum;
        for (int j = 0; j < pow(2, i - 1); j++) {
            u[j] = u[j * 2] + u[j * 2 + 1];
        }
    }

    return result;
}

vector<int> rsr_pp_inference(vector<int> v, const vector<vector<int>>& permutations, const vector<vector<int>>& segments, int k) {
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
    vector<int> partial_result;
    for (int i = 0; i < us.size(); i++) {
        // partial_result = vectorMatrixMultiply(us[i], bin_k);
        partial_result = step_three(us[i], k);
        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_result[j];
        }
    }
    return result;
}