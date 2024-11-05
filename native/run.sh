#!/bin/bash
clang++ $(find . -name '*.cpp' ! -name 'test.cpp') -o main
./main
rm main
