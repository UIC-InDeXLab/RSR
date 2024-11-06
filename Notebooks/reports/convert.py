import re
import csv
import sys

# Get filenames from command-line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <input_log_file> <output_csv_file>")
    sys.exit(1)

input_log_file = sys.argv[1]
output_csv_file = sys.argv[2]

# Regular expressions to capture log_n, k, RSR time, and RSRPP time
log_n_pattern = re.compile(r"log_n = (\d+)")
k_pattern = re.compile(r"k = (\d+)")
rsr_time_pattern = re.compile(r"RSR\|Time: (\d+)")
rsrpp_time_pattern = re.compile(r"RSRPP\|Time: (\d+)")

# Prepare a list to collect data
data = []

# Parse the log file
with open(input_log_file, 'r') as f:
    lines = f.readlines()

current_log_n = None
current_k = None
current_rsr_time = None
current_rsrpp_time = None

for line in lines:
    # Match log_n
    log_n_match = log_n_pattern.search(line)
    if log_n_match:
        current_log_n = 2 ** int(log_n_match.group(1))
        continue
    
    # Match k
    k_match = k_pattern.search(line)
    if k_match:
        current_k = int(k_match.group(1))
        continue
    
    # Match RSRPP time
    rsrpp_time_match = rsrpp_time_pattern.search(line)
    if rsrpp_time_match:
        current_rsrpp_time = int(rsrpp_time_match.group(1))
        continue

    # Match RSR time and append the row to data
    rsr_time_match = rsr_time_pattern.search(line)
    if rsr_time_match:
        current_rsr_time = int(rsr_time_match.group(1))
        if current_log_n and current_k:
            data.append([current_log_n, current_k, current_rsr_time, current_rsrpp_time])

# Write to CSV
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['n', 'k', 'rsr', 'rsrpp'])
    csvwriter.writerows(data)

print(f"Data written to {output_csv_file}")
