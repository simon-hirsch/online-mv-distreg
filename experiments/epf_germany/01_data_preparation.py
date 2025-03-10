import datetime

import numba as nb
import numpy as np
import pandas as pd


def collapse_columns(df: pd.DataFrame, sep: str = "_"):
    df.columns = [sep.join(map(str, col)).strip() for col in df.columns.values]
    return df


@nb.jit()
def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))


@nb.jit()
def average_absolute_deviation(x):
    return np.mean(np.abs(x - np.mean(x)))


@nb.jit()
def signed_co_movement(x, y):
    a = x - np.mean(x)
    b = y - np.mean(y)
    cov = np.mean(a * b)
    return np.sign(cov) * np.sqrt(np.abs(cov))


def get_co_standard_deviation(prices, window=7):
    H = prices.shape[1]
    rolled = prices.rolling(window, center=False).cov()
    rolled = np.sign(rolled) * np.abs(rolled) ** 0.5
    rolled = rolled.unstack()

    remove = rolled.columns.get_level_values(0) == rolled.columns.get_level_values(1)
    rolled = rolled.loc[:, ~remove]

    combos = [f"price_h{i}_h{j}_costd" for i in range(H) for j in range(i + 1, H)]

    c0 = rolled.columns.get_level_values(0).str.removesuffix("_price")
    c1 = rolled.columns.get_level_values(1).str.removesuffix("_price")

    rolled.columns = "price_" + c0 + "_" + c1 + "_costd"
    return rolled.loc[:, combos]


END_TRAIN = pd.to_datetime(datetime.date(2018, 12, 26))
DAILY_COLUMNS = ["eua", "api2_coal", "ttf_gas", "brent_oil"]
RENAME = {"renewables": "res"}

H = 24  # Hours
L = 7  # Max Lags

# data = pd.read_csv("de_prices.csv", index_col=0)
data = pd.read_csv(
    "https://raw.githubusercontent.com/gmarcjasz/distributionalnn/refs/heads/main/Datasets/DE.csv",
    index_col=0,
)

data.columns = data.columns.str.lower().str.replace("_da_forecast", "")
data = data.rename(columns=RENAME)
data.index = pd.to_datetime(data.index)

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


# Set-up daily data
# This data is the same for the whole day, hence we need to calculate it only once
# - Fundamental Fuel Prices
# - Mean Renewables generation and Load
# - Summary statistics for Lag-1 Data
daily = data.loc[:, DAILY_COLUMNS].resample("1D").mean().add_prefix("day_").shift(2)
daily["day_res"] = df.loc[:, "res"].mean(axis=1)
daily["day_load"] = df.loc[:, "load"].mean(axis=1)
daily["day_p_l1_min"] = df["price"].shift(1).min(axis=1)
daily["day_p_l1_max"] = df["price"].shift(1).max(axis=1)
daily["day_p_l1_avg"] = df["price"].shift(1).mean(axis=1)
daily["day_p_l1_q10"] = df["price"].shift(1).quantile(0.1, axis=1)
daily["day_p_l1_q90"] = df["price"].shift(1).quantile(0.9, axis=1)


# Create the lagged prices
prices = collapse_columns(df[["price"]].swaplevel(axis=1), "_").add_prefix("h")
prices_lagged = pd.concat(
    [prices.shift(i).add_suffix(f"_l{i}") for i in range(1, L + 1)], axis=1
)
prices_rolling_mean = (
    prices.rolling(7, center=False).mean().add_suffix("_roll7_avg").shift(1)
)
prices_rolling_std = (
    prices.rolling(7, center=False).std().shift(1).add_suffix("_roll7_std")
)
prices_rolling_costd = get_co_standard_deviation(prices=prices, window=7).shift(1)

# Set up renewable energy forecasts
load = collapse_columns(df[["load"]].swaplevel(axis=1), "_").add_prefix("h")
renewables = collapse_columns(df[["res"]].swaplevel(axis=1), "_").add_prefix("h")

# Combine everything
X = pd.concat(
    [
        renewables,
        load,
        prices_lagged,
        prices_rolling_mean,
        prices_rolling_std,
        prices_rolling_costd,
        daily,
        dummies,
    ],
    axis=1,
)

# Create training index
IDX_TRAIN = df.index <= END_TRAIN
IDX_TEST = ~IDX_TRAIN

X.loc[IDX_TRAIN, "flag"] = "train"
X.loc[IDX_TEST, "flag"] = "test"

# Drop data from NaN because of lagged variable generation
X = X.dropna(how="any", axis=0)
y = prices

available_index = X.index.intersection(y.index)

X = X.loc[available_index, :]  # .round(2)
y = y.loc[available_index, :]  # .round(2)

# Save to file
X.to_csv("prepared_x.csv")
y.to_csv("prepared_y.csv")
