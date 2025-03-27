#include <utility>
#include <vector>

using namespace std;

pair<vector<vector<int>>, vector<vector<int>>> preprocess(vector<vector<int>>& mat, int k);
vector<int> rsr_inference(vector<int> v, const vector<vector<int>>& permutations, const vector<vector<int>>& segments, vector<vector<int>> bin_k, int k);