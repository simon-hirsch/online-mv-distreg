import os

import numba as nb
import numpy as np
import pandas as pd

from . import FOLDER_DATA


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


def get_batch_scaled_dataset(scaling_idx):

    DAILY_COLUMNS = ["eua", "api2_coal", "ttf_gas", "brent_oil"]
    RENAME = {"renewables": "res"}
    L = 7  # Max Lags

    data = pd.read_csv(os.path.join(FOLDER_DATA, "de_prices_long.csv"), index_col=0)
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

    # Create the lagged prices
    # Do the batch scaling here
    # keep in mind pandas uses ddof=1 per default, so we need to change that to 0
    prices = collapse_columns(df[["price"]].swaplevel(axis=1), "_").add_prefix("h")
    prices_mean = prices.loc[scaling_idx, :].mean(axis=0)
    prices_std = prices.loc[scaling_idx, :].std(axis=0, ddof=0)
    prices = (prices - prices_mean) / prices_std

    prices_lagged = pd.concat(
        [prices.shift(i).add_suffix(f"_l{i}") for i in range(1, L + 1)], axis=1
    )
    prices_rolling_mean = (
        prices.rolling(7, center=False).mean().add_suffix("_roll7_avg").shift(1)
    )
    prices_rolling_std = (
        prices.rolling(7, center=False).std().shift(1).add_suffix("_roll7_std")
    )
    prices_rolling_invstd = 1 / prices.rolling(7, center=False).std().shift(
        1
    ).add_suffix("_roll7_invstd")
    prices_rolling_costd = get_co_standard_deviation(prices=prices, window=7).shift(1)

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

    # Set up renewable energy forecasts
    load = collapse_columns(df[["load"]].swaplevel(axis=1), "_").add_prefix("h")
    renewables = collapse_columns(df[["res"]].swaplevel(axis=1), "_").add_prefix("h")
    residload = load.sub(renewables.values, axis=1)
    residload.columns = residload.columns.str.replace("load", "residload")
    residload["day_residload"] = residload.mean(axis=1)

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

    # Drop data from NaN because of lagged variable generation
    X = X.dropna(how="any", axis=0)
    available_index = X.index.intersection(prices.index)
    X = X.loc[available_index, :]  # .round(2)
    y = prices.loc[available_index, :]  # .round(2)
    return X, y, prices_mean, prices_std
