#include <iostream>
#include <vector>
#include <cmath>
#include "naive.h"
#include "rsr.h"
#include "rsrpp.h"
#include "utils.h"

using namespace std;

int main() {
    for (int i = 2; i <= 12; i++) {
        int n = pow(2, i);
        cout << "n = " << n << endl;
        int k = static_cast<int>(ceil(log2(n) - log2(log2(n))));

        vector<vector<int>> mat = generateBinaryRandomMatrix(n);
        vector<int> v = generateRandomVector(n);
        vector<vector<int>> bin_k = generateBinaryMatrix(k);
        
        cout << "preprocessing..." << endl;
        auto per_segs = preprocess(mat, k);

        cout << "inference..." << endl;

        vector<int> actual_1 = rsr_pp_inference(v, per_segs.first, per_segs.second, k);
        vector<int> actual_2 = rsr_inference(v, per_segs.first, per_segs.second, bin_k, k);
        vector<int> expected = vectorMatrixMultiply(v, mat);

        for(int i = 0; i < n; i++) {
            if (actual_1[i] != expected[i]) {
                cout << "ERROR" << endl;
                return 1;
            }
            if (actual_2[i] != expected[i]) {
                cout << "ERROR" << endl;
                return 1;
            }
        }
    }
    cout << "PASSED" << endl;

    return 0;
}