#!/bin/bash

# Step 1: Check if build/ exists
if [ ! -d build ]; then
    echo "Creating build/ directory..."
    mkdir build
else
    echo "build/ directory already exists."
fi

# Step 2: Run cmake inside build/
cd build
cmake ..
make