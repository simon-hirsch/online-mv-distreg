import os
import subprocess

from const_and_helper import FOLDER, FOLDER_FIGURES, FOLDER_RESULTS, FOLDER_TABLES


def make_folder(path: str):
    try:
        os.makedirs(path, exist_ok=False)
    except OSError:
        print(path, "exists")
        print(os.listdir(path))


def run_file(file: str, folder: str = FOLDER):
    logfile = os.path.join(folder, "log_" + file.replace(".py", ".log"))
    with open(logfile, "w") as l:
        result = subprocess.run(
            ["python", os.path.join(folder, file)],
            stderr=l,
            stdout=l,
        )
    if result.returncode != 0:
        print(f"File {file} did not run successfully.")
    else:
        print(f"Successfully ran {file}.")


# Check that the data file is present
# If you want to use the shorter file, change this path accordingly
data_file = os.path.join(FOLDER, "de_prices_long.csv")
if not os.path.isfile(data_file):
    raise FileNotFoundError(f"Required data file not found: {data_file}")

# Make the folders
make_folder(FOLDER_FIGURES)
make_folder(FOLDER_RESULTS)
make_folder(FOLDER_TABLES)

# Run all files sequentially
# This is the main study
run_file("01_data_preparation.py")
run_file("02_benchmark_univariate.py")
run_file("02_benchmark_copula.py")
run_file("02_benchmark_conformal.py")
run_file("02_benchmark_garch.py")
run_file("03_multivariate_online_distreg.py")
run_file("05_calculate_scores.py")
run_file("06_make_figures.py")
run_file("06_make_table_error_scores.py")
run_file("06_make_table_timings.py")

# Additional experiments
run_file("04_multivariate_online_ablation.py")
run_file("04_multivariate_overfitting.py")
run_file("04_multivariate_online_batch.py")

print("Successfully finished simulation and created all reproduction files.")
