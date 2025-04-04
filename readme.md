# Regularized Online Multivariate Distributional Regression

This repo holds the experiments for the Paper _[Online Multivariate Regularized Distributional Regression for High-dimensional Probabilistic Electricity Price Forecasting
(Link to Arxiv)](https://arxiv.org/abs/2504.02518)_.

## Requirements

The following packages are needed to run the experiments

- scipy
- numba
- numpy
- pandas
- scikit-learn
- rolch
- scoringrules
- matplotlib
- seaborn
- autograd (for testing)

## Data

The data in `experiments/epf_germany` is taken from [Marcjasz, Grzegorz, et al. "Distributional neural networks for electricity price forecasting." Energy Economics 125 (2023)](https://www.sciencedirect.com/science/article/pii/S0140988323003419?casa_token=l42k_WCgotYAAAAA:ee4hs1n7VyZDJlczYjv9Ja86pdcpZJ19K-tToJc7WEX-KxNOmk3GS_gG2qfmOrlk7h2vQAx2uf1R) and the according [Github repository](https://github.com/gmarcjasz/distributionalnn).

## Source code

Please ensure you can import the code from `/src`. This is done in the experiment code by temporarily appending `"..\..\..\online_mv_distreg"` to the `PATH` for some of the `Python` files. If you're on a Windows machine, you might need to adjust this code. The code therein holds the estimator classes for the multivariate online distributional regression model.

## Experiment

The forecasting study is in `experiments/epf_germany`. Please run the file `00_run_study.py` to run all experiments in one go.
If you run the files separately, please keep the order for files 01 to 05 inclusive.

## Acknowlegdements

Simon Hirsch is employed as an industrial PhD student by Statkraft Trading GmbH and gratefully acknowledges the support and funding received. This work contains the author’s opinions and does not necessarily reflect Statkraft’s position.
