## Requirements
In order to run the `c++` code, you need `clang++` installed.

## Run Time Comparison
To get the report on time comparison for different values `n`, use `./run_time_compare.sh [algorithm]`. For `algorithm` use either of `naive`, `rsr`, or `rsrpp`.

## K Optmization
To run the `k` optmization code, run `./run_k_optmization.sh`. This code checks the running time of different values `k` for a specific `n` hard-coded in `k_optmization.cpp`.

## Run Tests
There are some tests written to verify the correctness of the algorithms. In order to run them use `./run_test.sh`