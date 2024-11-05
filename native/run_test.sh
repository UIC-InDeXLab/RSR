#!/bin/bash
clang++ $(find . -name '*.cpp' ! -name 'main.cpp') -o test
./test
rm test