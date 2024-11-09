#!/bin/bash
clang++ $(find . -name '*.cpp' ! -name 'test.cpp' ! -name 'k_optimization.cpp' ) -o time_compare
./time_compare $1
rm time_compare