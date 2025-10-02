# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
# Import packages
import datetime

import numpy as np
import pandas as pd
from const_and_helper import FOLDER_DATA, FORGET
from const_and_helper.data_prep import collapse_columns, get_co_standard_deviation
from ondil.scaler import OnlineScaler
from sklearn.preprocessing import SplineTransformer

# %%

END_TRAIN = pd.to_datetime(datetime.date(2018, 12, 26))
DAILY_COLUMNS = ["eua", "api2_coal", "ttf_gas", "brent_oil"]
RENAME = {"renewables": "res"}

H = 24  # Hours
L = 7  # Max Lags

data = pd.read_csv(os.path.join(FOLDER_DATA, "de_prices_long.csv"), index_col=0)
# data = pd.read_csv(
#     "https://raw.githubusercontent.com/gmarcjasz/distributionalnn/refs/heads/main/Datasets/DE.csv",
#     index_col=0,
# )

data.columns = data.columns.str.lower().str.replace("_da_forecast", "")
data = data.rename(columns=RENAME)
data.index = pd.to_datetime(data.index)

# %%
data["date"] = data.index.date
data["hour"] = data.index.hour
data["wd"] = data.index.day_name()
data["wd"] = data.wd.str.lower().map(lambda x: x[:3])

dummies = pd.get_dummies(data=data["wd"], columns=["wd"], prefix="binary")
dummies = dummies.resample("1D").mean().astype(int)
dummies = dummies.drop(["binary_wed"], axis=1)  # Normalize to wednesday

df = data.set_index(["date", "hour"]).drop(["wd"] + DAILY_COLUMNS, axis=1)
df = df.unstack("hour")
df.index = pd.to_datetime(df.index)

# Create training index
IDX_TRAIN = df.index <= END_TRAIN
IDX_TEST = ~IDX_TRAIN


# %%
# Preprocess the price data
# Run mean-variance scaling on the prices
# Box-Cox transformation
scaler = OnlineScaler(forget=FORGET)

# Create the lagged prices
prices = collapse_columns(df[["price"]].swaplevel(axis=1), "_").add_prefix("h")
prices_untransformed = prices.copy()
prices_means = prices.copy()
prices_variances = prices.copy()

# %%
scaler.fit(prices.loc[IDX_TRAIN, :].values)
prices.loc[IDX_TRAIN, :] = scaler.transform(prices.loc[IDX_TRAIN, :].values)
prices_means.loc[IDX_TRAIN, :] = scaler.mean_
prices_variances.loc[IDX_TRAIN, :] = scaler.var_**0.5

# %%
for d in df.index[IDX_TEST]:
    scaler.update(prices.loc[[d], :].values)
    prices.loc[d, :] = scaler.transform(prices.loc[[d], :].values)
    prices_means.loc[d, :] = scaler.mean_
    prices_variances.loc[d, :] = scaler.var_**0.5

# prices = boxcox(prices, power=0.5, shift=1.0)

# %%
prices_lagged = pd.concat(
    [prices.shift(i).add_suffix(f"_l{i}") for i in range(1, L + 1)], axis=1
)
prices_rolling_mean = (
    prices.rolling(7, center=False).mean().add_suffix("_roll7_avg").shift(1)
)
prices_rolling_std = (
    prices.rolling(7, center=False).std().shift(1).add_suffix("_roll7_std")
)
prices_rolling_invstd = 1 / prices.rolling(7, center=False).std().shift(1).add_suffix(
    "_roll7_invstd"
)
prices_rolling_costd = get_co_standard_deviation(prices=prices, window=7).shift(1)

# %%
# Set-up daily data
# This data is the same for the whole day, hence we need to calculate it only once
# - Fundamental Fuel Prices
# - Mean Renewables generation and Load
# - Summary statistics for Lag-1 Data
daily = data.loc[:, DAILY_COLUMNS].resample("1D").mean().add_prefix("day_").shift(2)
daily = np.log(daily)
daily["day_price_l1_min"] = prices.shift(1).min(axis=1)
daily["day_price_l1_max"] = prices.shift(1).max(axis=1)
daily["day_price_l1_avg"] = prices.shift(1).mean(axis=1)
daily["day_price_l1_q10"] = prices.shift(1).quantile(0.1, axis=1)
daily["day_price_l1_q90"] = prices.shift(1).quantile(0.9, axis=1)


# %%
# Set up renewable energy forecasts
load = collapse_columns(df[["load"]].swaplevel(axis=1), "_").add_prefix("h")
renewables = collapse_columns(df[["res"]].swaplevel(axis=1), "_").add_prefix("h")

residload = load.sub(renewables.values, axis=1)
residload.columns = residload.columns.str.replace("load", "residload")
residload["day_residload"] = residload.mean(axis=1)


# %%

# Combine everything
X = pd.concat(
    [
        residload.add_suffix("_linear"),
        prices_lagged,
        prices_rolling_mean,
        prices_rolling_std,
        prices_rolling_invstd,
        prices_rolling_costd,
        daily,
        dummies,
    ],
    axis=1,
)


X.loc[IDX_TRAIN, "flag"] = "train"
X.loc[IDX_TEST, "flag"] = "test"

# Drop data from NaN because of lagged variable generation
X = X.dropna(how="any", axis=0)
available_index = X.index.intersection(prices.index)

X = X.loc[available_index, :]  # .round(2)
y = prices.loc[available_index, :]  # .round(2)

prices_untransformed.loc[available_index, :].to_csv(
    os.path.join(FOLDER_DATA, "prices_untransformed.csv")
)

prices_means.loc[available_index, :].to_csv(
    os.path.join(FOLDER_DATA, "prices_means.csv")
)
prices_variances.loc[available_index, :].to_csv(
    os.path.join(FOLDER_DATA, "prices_variances.csv")
)

# Save to file
X.to_csv(os.path.join(FOLDER_DATA, "prepared_long_x.csv"))
y.to_csv(os.path.join(FOLDER_DATA, "prepared_long_y.csv"))

# %%

# Calculate marginal costs based on the fuel prices
# This is a simple merit-order model approach, where beta is the efficiency factor
# We multiply this by the spline of the residual load, since different plants
# Run at the different residual load levels.

daily["margcost_gas"] = np.log(daily["day_ttf_gas"]) + 0.2 * np.log(daily["day_eua"])
daily["margcost_coal"] = np.log(daily["day_api2_coal"]) + 0.35 * np.log(
    daily["day_eua"]
)
daily["margcost_oil"] = np.log(daily["day_brent_oil"]) + 0.3 * np.log(daily["day_eua"])

basis = []
for column in residload.columns:
    training = residload.loc[residload.index <= END_TRAIN, [column]]
    transformer = SplineTransformer(
        include_bias=False,
        n_knots=4,
        degree=2,
        extrapolation="constant",
    )
    transformer.fit(training)
    out = transformer.transform(residload.loc[:, [column]])
    out = pd.DataFrame(out, columns=np.arange(out.shape[1]).astype(str))
    out = out.add_prefix("_basis")
    out = out.add_prefix(column)
    basis.append(out)
    for tech in ["gas", "coal", "oil"]:
        tech_basis = out.mul(daily[f"margcost_{tech}"].values, axis=0)
        tech_basis.columns = tech_basis.columns.str.replace("basis", f"{tech}_basis")
        basis.append(tech_basis)

basis = pd.concat(basis, axis=1)
basis = basis.set_index(residload.index)

prices_inv_rolling_std = 1 / prices_rolling_std
prices_inv_rolling_std.columns = prices_inv_rolling_std.columns.str.replace(
    "std", "invstd"
)

# %%
# Combine everything
X_spline = pd.concat(
    [
        residload.add_suffix("_linear"),
        basis,
        prices_lagged,
        prices_rolling_mean,
        prices_rolling_std,
        prices_rolling_invstd,
        prices_inv_rolling_std,
        prices_rolling_costd,
        daily,  # since marginal costs are now in the basis
        dummies,
    ],
    axis=1,
)

# Create training index
IDX_TRAIN = df.index <= END_TRAIN
IDX_TEST = ~IDX_TRAIN

X_spline.loc[IDX_TRAIN, "flag"] = "train"
X_spline.loc[IDX_TEST, "flag"] = "test"

# Drop data from NaN because of lagged variable generation
X_spline = X_spline.dropna(how="any", axis=0)

available_index = X_spline.index.intersection(y.index)

X_spline = X_spline.loc[available_index, :]  # .round(2)
y = y.loc[available_index, :]  # .round(2)

# Save to file
X_spline.to_csv(os.path.join(FOLDER_DATA, "prepared_long_x_spline.csv"))
y.to_csv(os.path.join(FOLDER_DATA, "prepared_long_y_spline.csv"))
