#!/bin/bash
clang++ $(find . -name '*.cpp' ! -name 'test.cpp' ! -name 'time_compare.cpp' ) -o k_opt
./k_opt
rm k_opt