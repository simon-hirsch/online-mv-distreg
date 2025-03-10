from abc import ABC
from typing import Literal

import numpy as np
from rolch.utils import (
    calculate_asymptotic_training_length,
    calculate_effective_training_length,
)


class BaseOnlineScaler(ABC):

    forget: float
    m: np.ndarray
    v: np.ndarray
    n_observations: int
    n_effective: float
    n_asymptotic: float
    shape: Literal["flat", "cube"]

    @property
    def is_fitted(self) -> bool:
        return self.n_observations > 0

    @property
    def var(self) -> np.ndarray:
        return self.v

    @property
    def avg(self) -> np.ndarray:
        return self.m

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.v)


class MultivariateOnlineScaler(BaseOnlineScaler):
    def __init__(
        self,
        # to_scale: np.ndarray,
        forget: float = 0.0,
        do_scale: bool = True,
    ):
        self.do_scale = do_scale
        self.forget = forget
        self.n_observations = 0
        self.n_asymptotic = 0
        self.n_effective = 0

    def _infer_shape(self, x: np.ndarray) -> None:
        if len(x.shape) == 2:
            self.shape = "flat"
        elif len(x.shape) == 3:
            self.shape = "cube"
        else:
            raise ValueError("Do not support passed shape.")

    def fit(self, x: np.ndarray, to_scale: np.ndarray) -> None:
        if self.do_scale:
            self.to_scale = to_scale
            self.n_observations = x.shape[0]
            self.n_asymptotic = calculate_asymptotic_training_length(self.forget)
            self._infer_shape(x=x)
            # TODO: This seems unncessary
            if self.shape == "flat":
                self.m = np.mean(x[:, self.to_scale], axis=0)
                self.v = np.var(x[:, self.to_scale], axis=0)
            elif self.do_scale and self.shape == "cube":
                self.m = np.mean(x[..., self.to_scale], axis=0)
                self.v = np.var(x[..., self.to_scale], axis=0)
            self.M = self.v * self.n_observations
            self.fitted = True
        else:
            pass

    def partial_fit(self, x: np.ndarray) -> None:
        if self.do_scale and not self.fitted:
            raise ValueError("Cannot partial_fit() an object before the initial fit.")
        if self.do_scale:
            # Loop over all observations of x
            for i in range(x.shape[0]):
                # np.array[some_index, ..., some_index] indexes on the first and last axis no matter
                # the shape in between
                self.n_observations += 1
                self.n_effective = calculate_effective_training_length(
                    self.forget, self.n_observations
                )

                forget_scaled = self.forget * np.maximum(
                    self.n_asymptotic / self.n_effective, 1.0
                )
                # The advanced indexing leads to an implicit transpose
                # So we need to transpose back
                diff = x[[i], ..., self.to_scale].T - self.m
                incr = forget_scaled * diff

                if forget_scaled > 0:
                    self.m += incr
                    self.v = (1 - forget_scaled) * (self.v + forget_scaled * diff**2)
                else:
                    self.m += diff / self.n_observations
                    self.M += diff * (x[[i], ..., self.to_scale].T - self.m)
                    self.v = self.M / self.n_observations
        else:
            pass

    def update(self, x: np.ndarray) -> None:
        """Update the scaler. Alias for partial_fit()

        Args:
            x (np.ndarray): New data $X$.
        """

        self.partial_fit(x=x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.copy(x)
        if self.do_scale:
            out[..., self.to_scale] = (
                x[..., self.to_scale] - self.m[np.newaxis, ...]
            ) / np.sqrt(self.v[np.newaxis, ...])
            return out
        else:
            return x

    def fit_transform(self, x: np.ndarray, to_scale: np.ndarray) -> np.ndarray:
        self.fit(x=x, to_scale=to_scale)
        out = self.transform(x)
        return out

    def partial_fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.partial_fit(x)
        out = self.transform(x)
        return out

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        zero_variance = np.isclose(self.var, 0)
        if np.any(zero_variance):
            print("Zero variance input in X. Please check your design matrix.")
        out = np.copy(x)
        if self.do_scale:
            out[..., self.to_scale] = x[..., self.to_scale] * np.sqrt(
                np.expand_dims(self.v, 0)
            ) + np.expand_dims(self.m, 0)
            return out
        else:
            return x


class OnlineNormalizer(BaseOnlineScaler):
    def __init__(
        self,
        # to_scale: np.ndarray,
        forget: float = 0.0,
        do_scale: bool = True,
    ):
        self.do_scale = do_scale
        self.forget = forget
        self.n_observations = 0
        self.n_asymptotic = 0
        self.n_effective = 0

    def _infer_shape(self, x: np.ndarray) -> None:
        if len(x.shape) == 2:
            self.shape = "flat"
        elif len(x.shape) == 3:
            self.shape = "cube"
        else:
            raise ValueError("Do not support passed shape.")

    def fit(self, x: np.ndarray, to_scale: np.ndarray) -> None:
        if self.do_scale:
            self.to_scale = to_scale
            self.n_observations = x.shape[0]
            self.n_asymptotic = calculate_asymptotic_training_length(self.forget)
            self._infer_shape(x=x)
            # TODO: This seems unncessary
            if self.shape == "flat":
                self.m = np.mean(x[:, self.to_scale], axis=0)
                self.v = np.var(x[:, self.to_scale], axis=0)
            elif self.do_scale and self.shape == "cube":
                self.m = np.mean(x[..., self.to_scale], axis=0)
                self.v = np.var(x[..., self.to_scale], axis=0)
            self.M = self.v * self.n_observations
            self.fitted = True
        else:
            pass

    def partial_fit(self, x: np.ndarray) -> None:
        if self.do_scale and not self.fitted:
            raise ValueError("Cannot partial_fit() an object before the initial fit.")
        if self.do_scale:
            # Loop over all observations of x
            for i in range(x.shape[0]):
                # np.array[some_index, ..., some_index] indexes on the first and last axis no matter
                # the shape in between
                self.n_observations += 1
                self.n_effective = calculate_effective_training_length(
                    self.forget, self.n_observations
                )

                forget_scaled = self.forget * np.maximum(
                    self.n_asymptotic / self.n_effective, 1.0
                )
                # The advanced indexing leads to an implicit transpose
                # So we need to transpose back
                diff = x[[i], ..., self.to_scale].T - self.m
                incr = forget_scaled * diff

                if forget_scaled > 0:
                    self.m += incr
                    self.v = (1 - forget_scaled) * (self.v + forget_scaled * diff**2)
                else:
                    self.m += diff / self.n_observations
                    self.M += diff * (x[[i], ..., self.to_scale].T - self.m)
                    self.v = self.M / self.n_observations
        else:
            pass

    def update(self, x: np.ndarray) -> None:
        """Update the scaler. Alias for partial_fit()

        Args:
            x (np.ndarray): New data $X$.
        """

        self.partial_fit(x=x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.copy(x)
        if self.do_scale:
            out[..., self.to_scale] = x[..., self.to_scale] / np.sqrt(
                self.v[np.newaxis, ...]
            )
            return out
        else:
            return x

    def fit_transform(self, x: np.ndarray, to_scale: np.ndarray) -> np.ndarray:
        self.fit(x=x, to_scale=to_scale)
        out = self.transform(x)
        return out

    def partial_fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.partial_fit(x)
        out = self.transform(x)
        return out

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        out = np.copy(x)
        if self.do_scale:
            out[..., self.to_scale] = x[..., self.to_scale] * np.sqrt(
                np.expand_dims(self.v, 0)
            )
            return out
        else:
            return x
