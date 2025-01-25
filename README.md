# 🔥 An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks

This repository contains code and experiments for the paper, [An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](https://arxiv.org/abs/2411.06360).

The codebase provides two sets of experiments: a NumPy-based implementation and native C++ implementations.

---

## 🧮 NumPy Implementations

The NumPy implementations of the matrix multipliers (`Naive`, `RSR`, and `RSR++`) are found in `numpy_impl` directory. You can use these multipliers by instantiating a `Multiplier` object and passing a weight matrix `A` (required) and an optional parameter `k`. Initialization automatically includes any necessary preprocessing steps, and you can perform inference on input vectors using the `multiply` method.

### ⚙️ Requirements
Ensure you have `Python >= 3.6` installed, along with all packages listed in `requirements.txt`.

### ✅ Testing the Multipliers
To validate the correctness of the `RSR` and `RSR++` multipliers, run `rsr_test.py`. This script randomly generates a weight matrix and an input vector, then compares the results of the multiplication with the ground truth. To run the tests use `./run_test.sh` inside `numpy_impl` directory.

---

## 💻 Native C++ Implementations

Native C++ implementations for the matrix multipliers are available in the `native` directory.

### ⚙️ Requirements
To compile and run the C++ code, you’ll need `clang++` installed.

### ⏱️ Run Time Comparison
To compare run times for different values of `n` across algorithms, use the script `./run_time_compare.sh [algorithm]`, where `[algorithm]` can be one of `naive`, `rsr`, or `rsrpp`.

### 🔧 `k` Optimization
To test various values of `k` for runtime optimization, run `./run_k_optimization.sh`. This script benchmarks the run times for different `k` values, with the target `n` value specified in `k_optimization.cpp`.

### 🧪 Running Tests
Several tests are provided to ensure algorithmic correctness. Run these tests by executing `./run_test.sh` inside `native` directory.

---

## 🔥 Torch Implementations
Torch implementations are inside `torch_impl` directory. This implementation works only with torch tensor operations. To run the tests and examples run `./run_tests.sh` inside `torch_impl` directory.

---
## 🧠 LLM Experiments
The LLM implementations are inside `llm` directory, which contains both codes for the inference on CPU and GPU in the corresponding dirs. The `patch.py` file contains patched functions for preprocessing and the new `forward` function for the `BitLinear` module. The notebook `cpu/profile.ipynb` contains an example code to apply patch and profile the running time.
