import os
import subprocess

# Make the folders
try:
    os.makedirs("/figues", exist_ok=False)
except OSError:
    print("/figures folder exists")

try:
    os.makedirs("/results", exist_ok=False)
except OSError:
    print("/results folder exists")

try:
    os.makedirs("/tables", exist_ok=False)
except OSError:
    print("/tables folder exists")

# Run all files sequentially
subprocess.run(["python", "01_data_preparation.py"])
subprocess.run(["python", "02_univariate_benchmarks.py"])
subprocess.run(["python", "03_copula_model.py"])
subprocess.run(["python", "04_multivariate_online_distreg.py"])
subprocess.run(["python", "05_calculate_scores.py"])
subprocess.run(["python", "06_make_figures.py"])
subprocess.run(["python", "07_make_tables_error_scores.py"])
subprocess.run(["python", "08_make_table_timings.py"])

print("Successfully finished simulation and created all reproduction files.")
