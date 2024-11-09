#!/bin/bash
clang++ $(find . -name '*.cpp' ! -name 'main.cpp' ! -name 'k_optimization.cpp' ! -name 'time_compare.cpp') -o test
./test
rm test