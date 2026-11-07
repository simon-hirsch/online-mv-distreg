# Regularized Online Multivariate Distributional Regression

This repo holds the experiments for the Paper _[Online Multivariate Regularized Distributional Regression for High-dimensional Probabilistic Electricity Price Forecasting
(Link to Arxiv)](https://arxiv.org/abs/2504.02518)_.

This reproduction package has been updated on Github throughout the revision process. The last update was on Oct 31, 2026 to extend the documentation of the reproduction package.

For any questions, please contact Simon Hirsch under the e-mail address noted in the linked paper. 

## Data

The data used in the revised version of the paper is not included in this repository as it cannot be shared publicly. The author gratefully acknowedges Arkadiusz Lipiecki, Bartosz Uniejewski, Rafał Weron for sharing the data and allowing its use in this research. Please contact the authors of [Lipiecki et al. "Postprocessing of point predictions for probabilistic forecasting of day-ahead electricity prices: The benefits of using isotonic distributional regression". Energy Economics 139 (2024)](https://www.sciencedirect.com/science/article/pii/S014098832400642X) for access to the data.

A similar, but shorter dataset can be taken from [Marcjasz, Grzegorz, et al. "Distributional neural networks for electricity price forecasting." Energy Economics 125 (2023)](https://www.sciencedirect.com/science/article/pii/S0140988323003419?casa_token=l42k_WCgotYAAAAA:ee4hs1n7VyZDJlczYjv9Ja86pdcpZJ19K-tToJc7WEX-KxNOmk3GS_gG2qfmOrlk7h2vQAx2uf1R) and the according [Github repository](https://github.com/gmarcjasz/distributionalnn). The differences between both data sets are discussed in Lipiecki et al. (2024). The initial version of this paper was based on this data set and the results qualitatively hold for both data sets.

Please name the data set and place it in this folder:

```bash
/experiments/epf_germany/de_price_long.csv
```

and ensure it has the following columns:

- [empty]
- Price
- Load_DA_Forecast
- Renewables_DA_Forecast
- EUA
- API2_Coal
- TTF_Gas
- Brent_oil

so that the data preprocessing script works without changes.

## Requirements

The following packages are needed to run the experiments

- scipy
- numba
- numpy
- pandas
- scikit-learn
- ondil
- scoringrules
- matplotlib
- seaborn
- arch
- tqdm

## Reproduction steps

The file `run_study.sh` contains a bash script that creates a conda environment, installs the requirements and runs the experiments. Please adjust the name of the conda environment as needed. Please ensure that you have the data file available!

- Clone the repo.
- Download the data as described above and place it in the folder `data` (create it if necessary).
- Create a virtual environment and install the requirements. Requirements are provided in `req.txt` in combination with python 3.10.
- From the top-level folder, run `python experiments/epf_germany/00_run_study.py`.

## Scripts and results

In the `const_and_helpers/__init__.py` we define the following places to store (intermediate) results, figures, tables. These folders are created in the `00_run_study.py` if not created before.

``` 
FOLDER = "experiments/epf_germany/"
FOLDER_RESULTS = "experiments/epf_germany/results_revision/"
FOLDER_DATA = "experiments/epf_germany/"
FOLDER_FIGURES = "experiments/epf_germany/figures/"
FOLDER_TABLES = "experiments/epf_germany/tables/"
```

### Main Scripts

Running the main scripts on a [standard laptop](#compute-setup) should not take longer than a few hours (in total). The estimation files run some basic logging into files called `log_{scriptname}.log`.

- `01_data_preparation.py` creates the prepared data sets for $X$ and $y$ called `prepared_long_x.py` and `prepared_long_y.py` and the runs the preparation of the incrementally scaled prices, their incremental means and variances.
- `02_benchmark_xxx.py` run the benchmark models. For the benchmark models, we store `pred_xxx.npz` files with the predictions (distribution parameters) and `sims_xxx.npz` files with the actual simulations 
- `03_multivariate_online_distreg.py` runs the online study and again stores a file of predictions and simulations.
- `05_calculate_scores.py` takes the `sims_xxx.npz` files and calculates raw scores.
- `06_make_xxx.py` these three scripts will make the figures and tables with scores and the tables and figure with timings. This will create all tables and figures in one go.

### Additional analyses

Here, the `04_multivariate_online_batch.py` is computationally intensive and runs ~1-2 days on a laptop.

- `04_multivariate_online_batch.py` runs the study comparing online and batch computational efficiency (Section 4.3)
- `04_multivariate_online_ablation.py` runs the ablation study on the regularization effect and stores the according figures (Appendix A)
- `04_multivariate_overfitting.py` runs the 8-fold CV study on the model selection (Appendix A)
- `05_additional_calibration.py` runs the calibration analysis for price spike and high-spread periods (Supplement Section 4).

### Mapping figures and scripts

We save all figures as PDF and PNG files for presentations and the paper. We save tables as tex and html.

#### Main Text

| Figure / Table    | Script                                | File                                  | Comment |
|---                |---                                    |----                                   | ----    |
| Figure 1          | `06_make_figures.py`                  | `correlation.pdf`                     |  |
| Figure 5          | `06_make_figures.py`                  | `evoluation_cov_and_corr.pdf`         |  |
| Figure 6          | `05_additional_calibration.py`        | `calibration_prediction_bands.pdf`    |  |
| Table 4           | `06_make_table_error_scores.py`       | `scoringrules_{a,b,c}.tex`            |  |
| Figure 7          | `06_make_table_error_scores.py`       | `skill_scores_timeseries.pdf`         |  |
| Table 5           | `06_make_table_error_scores.py`       | `spike_analysis_{main, app}.pdf`      | Also creates Table 1 in the supplement. |
| Table 6           | `06_make_table_timings.py`            | `timings.tex`                         | Combined Figure/Table. |
| Table 6           | `06_make_table_timings.py`            | `computation_times.pdf`               | Combined Figure/Table. |
| Figure 8          | `04_multivariate_online_batch.py`     | `eff_frontier.pdf`                    | Long runtime (see above). |
| Figure A.9          | `04_multivariate_online_ablation.py`        | `ablation_regularization.pdf`         | Appendix A. Combined Figure. |
| Figure A.9          | `04_multivariate_overfitting.py`            | `overfitting.pdf`                     | Appendix A. Combined Figure. |
| Figure B.1          | `06_make_figures.py`                        | `dm_test.pdf`                         | Appendix B. |

#### Supplement

| Figure / Table    | Script                                | File                                  | Comment |
|---                |---                                    |----                                   | ----    |
| Table 1           | `06_make_table_error_scores.py`       | `spike_analysis_{main, app}.pdf`      | 
| Figure 1          | `05_additional_calibration.py`        | `calibration_comparison_spikes.pdf`   |  |
| Figure 2          | `05_additional_calibration.py`        | `calibration_comparison_spreads.pdf`  |  |

## Code in `src`

If you want to run individual files as notebooks, please ensure you can import the code from `/src`. This is done in the experiment code by the following line:

```python
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
```

which appends the working directory to the `PATH`. If you're on a Windows machine, you might need to adjust this code.

## Compute setup

The experiments in this paper have been carried out on a 2024 standard laptop equipped with an Intel Core i7 CPU (up to 4.9 Ghz, 24 MB cache) and 32 GB RAM, running Fedora as OS. The whole study requires roughly 15 GB of storage for data, intermediate and final results, figures and tables.

## Acknowledgements

Simon Hirsch is employed as an industrial PhD student by Statkraft Trading GmbH and gratefully acknowledges the support and funding received. This work contains the author’s opinions and does not necessarily reflect Statkraft’s position.
