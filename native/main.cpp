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

void run_time_rsr(int n, int k) {
    vector<int> result;
    vector<int> copied_v;
    vector<vector<int>> copied_mat;
    auto start = high_resolution_clock::now();
    auto end = start;
    int agg = 0;

    // Generate random
    vector<vector<int>> mat = generateBinaryRandomMatrix(n);
    vector<int> v = generateRandomVector(n);
    vector<vector<int>> bin_k = generateBinaryMatrix(k);

    // Preprocess
    cout << "Preprocessing..." << endl << flush;
    copied_mat = copy(mat);
    auto perm_seg = preprocess(copied_mat, k);

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

void run_time_rsrpp(int n, int k) {
    vector<int> result;
    vector<int> copied_v;
    vector<vector<int>> copied_mat;
    auto start = high_resolution_clock::now();
    auto end = start;
    int agg = 0;

    // Generate random
    vector<vector<int>> mat = generateBinaryRandomMatrix(n);
    vector<int> v = generateRandomVector(n);

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
}

void run_time_naive(int n) {
    vector<int> result;
    vector<int> copied_v;
    vector<vector<int>> copied_mat;
    auto start = high_resolution_clock::now();
    auto end = start;
    int agg = 0;

    // Generate random
    vector<vector<int>> mat = generateBinaryRandomMatrix(n);
    vector<int> v = generateRandomVector(n);

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

void compare_time(int n, int k) {
    vector<int> result;
    vector<int> copied_v;
    vector<vector<int>> copied_mat;
    auto start = high_resolution_clock::now();
    auto end = start;
    int agg = 0;

    cout << "log(N) = " << log2(n) << endl << flush;

    // Generate random
    vector<vector<int>> mat = generateBinaryRandomMatrix(n);
    vector<int> v = generateRandomVector(n);
    vector<vector<int>> bin_k = generateBinaryMatrix(k);

    // Preprocess
    cout << "Preprocessing..." << endl << flush;
    copied_mat = copy(mat);
    auto perm_seg = preprocess(copied_mat, k);

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
    vector<int> rsrpp_k = {5, 6, 8, 8, 9, 10};
    vector<int> rsr_k = {4, 4, 5, 6, 6, 6};

    string method = argv[1];

    for (int i = 0; i < log_ns.size(); i++) {
        cout << "log_n: " << log_ns[i] << endl;
        if (method == "rsr") {
            run_time_rsr(pow(2, log_ns[i]), rsr_k[i]);
        } else if (method == "rsrpp") {
            run_time_rsrpp(pow(2, log_ns[i]), rsr_k[i]);
        } else {
            run_time_naive(pow(2, log_ns[i]));
        }
    }

    // for (int i = 0; i < log_ns.size(); i++) {
    //     cout << "log_n = " << log_ns[i] << endl;
    //     compare_k(log_ns[i]);
    // }
    return 0;
}