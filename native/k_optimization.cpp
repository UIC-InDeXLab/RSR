#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.h"
#include "rsr.h"
#include "rsrpp.h"
#include "naive.h"

using namespace std;
using namespace std::chrono;

void compare_k(int log_n) {
    int n = pow(2, log_n);
    int k;
    vector<int> result;
    vector<int> copied_v;
    vector<vector<int>> copied_mat;
    auto start = high_resolution_clock::now();
    auto end = start;
    int agg = 0;

    for (int k = 1; k <= 12; k++) {
        cout << "k = " << k << endl << flush;

        // Generate random
        vector<vector<int>> mat = generateBinaryRandomMatrix(n);
        vector<int> v = generateRandomVector(n);
        vector<vector<int>> bin_k = generateBinaryMatrix(k);

        // Preprocess
        cout << "Preprocessing..." << endl << flush;
        copied_mat = copy(mat);
        auto perm_seg = preprocess(copied_mat, k);

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

        // RSR
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
    }
}

int main(int argc, char* argv[]) {
    vector<int> log_ns = {11, 12, 13, 14, 15, 16};

    for (int i = 0; i < log_ns.size(); i++) {
        cout << "log_n = " << log_ns[i] << endl;
        compare_k(log_ns[i]);
    }
    return 0;
}