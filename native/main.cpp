#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.h"
#include "rsr.h"
#include "rsrpp.h"
#include "naive.h"

using namespace std;
using namespace std::chrono;

vector<vector<int>> copy(const vector<vector<int>>& mat) {
    vector<vector<int>> copied(mat.size(), vector<int>(mat[0].size()));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[0].size(); j++) {
            copied[i][j] = mat[i][j];
        }
    }
    return copied;
}

vector<int> copy(const vector<int>& v) {
    vector<int> copied(v.size());

    for (int i = 0; i < v.size(); i++) {
        copied[i] = v[i];
    }
    
    return copied;
}

int main() {
    int n;
    int k;
    vector<int> result;
    vector<int> copied_v;
    vector<vector<int>> copied_mat;
    auto start = high_resolution_clock::now();
    auto end = start;
    int agg = 0;

    for (int log_n = 10; log_n <= 15; log_n++) {
        cout << "log(N) = " << log_n << endl << flush;

        // Init n, k
        n = pow(2, log_n);
        k = static_cast<int>(ceil(log2(n) - log2(log2(n))));

        // Generate random
        vector<vector<int>> mat = generateBinaryRandomMatrix(n);
        vector<int> v = generateRandomVector(n);
        vector<vector<int>> bin_k = generateBinaryMatrix(k);

        // RSR
        cout << "RSR|Preprocessing..." << endl << flush;
        copied_mat = copy(mat);
        auto perm_seg = preprocess(copied_mat, k);

        cout << "RSR|Inference" << flush;
        agg = 0;
        for (int j = 0; j < 10; j++) {
            copied_v = copy(v);
            start = high_resolution_clock::now();
            result = rsr_inference(copied_v, perm_seg.first, perm_seg.second, bin_k, k);
            end = high_resolution_clock::now();
            agg += duration_cast<milliseconds>(end - start).count();
            cout << "." << flush;
        }
        cout << endl << "RSR|Time: " << agg / 10 << endl << flush;

        // RSRPP
        cout << "RSRPP|Inference" << flush;
        agg = 0;
        for (int j = 0; j < 10; j++) {
            copied_v = copy(v);
            start = high_resolution_clock::now();
            result = rsr_pp_inference(copied_v, perm_seg.first, perm_seg.second, k);
            end = high_resolution_clock::now();
            agg += duration_cast<milliseconds>(end - start).count();
            cout << "." << flush;
        }
        cout << endl << "RSRPP|Time: " << agg / 10 << endl << flush;

        // Naive
        cout << "Naive|Multiplication" << flush;
        agg = 0;
        for (int j = 0; j < 10; j++) {
            start = high_resolution_clock::now();
            result = vectorMatrixMultiply(v, mat);
            end = high_resolution_clock::now();
            agg += duration_cast<milliseconds>(end - start).count();
            cout << ".";
        }
        cout << endl << "Naive|Time: " << agg / 10 << endl << flush;
    }

    return 0;
}