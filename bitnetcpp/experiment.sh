#!/bin/bash

sizes=(512 1024 2048 4096)
output_file="time_report.txt"

# If the file exists, remove it
rm -f "$output_file"

# Create a new empty file
touch "$output_file"

for a in "${sizes[@]}"; do
    echo "Running with size $a" >> "$output_file"
    ./build/main "$a" "$a" >> "$output_file"
    echo "" >> "$output_file"
done