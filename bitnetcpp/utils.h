#include <vector>

using namespace std;

void matrixMultiply(int N, const float *A, const float *B, float *C);
float *load_matrix_bin(const char *filename);
int binaryVectorToInt(const vector<int>& binaryVec);
vector<int> vectorMatrixMultiply(const vector<int>& vec, const vector<vector<int>>& mat);