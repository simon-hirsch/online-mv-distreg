from abc import ABC
from typing import Dict, Tuple, Union

import numpy as np
from rolch import (
    init_gram,
    init_inverted_gram,
    init_y_gram,
    update_gram,
    update_inverted_gram,
    update_y_gram,
)


class Estimator(ABC):
    """
    Base class for estimators.
    """

    def __init__(self, method: Dict[int, str]):
        self.method = method
        self.beta = {}
        self.beta_path = {}

    def make_gram(self, x: np.ndarray, w: np.ndarray, param: int) -> np.ndarray:
        """
        Make the Gram matrix.

        Args:
            x (np.ndarray): Covariate matrix.
            w (np.ndarray): Weight matrix.
            param (int): Parameter index.

        Returns:
            np.ndarray: Gram matrix.
        """
        if self.method[param] == "ols":
            return init_inverted_gram(X=x, w=w, forget=self.forget[param])
        elif self.method[param] == "lasso":
            return init_gram(X=x, w=w, forget=self.forget[param])

    def make_y_gram(
        self, x: np.ndarray, y: np.ndarray, w: np.ndarray, param: int
    ) -> np.ndarray:
        """
        Make the Gram matrix for the response variable.

        Args:
            x (np.ndarray): Covariate matrix.
            y (np.ndarray): Response variable.
            w (np.ndarray): Weight matrix.
            param (int): Parameter index.

        Returns:
            np.ndarray: Gram matrix for the response variable.
        """
        return init_y_gram(X=x, y=y, w=w, forget=self.forget[param])

    def update_gram(
        self, gram: np.ndarray, x: np.ndarray, w: np.ndarray, param: int
    ) -> np.ndarray:
        """
        Update the Gram matrix.

        Args:
            gram (np.ndarray): Current Gram matrix.
            x (np.ndarray): Covariate matrix.
            w (np.ndarray): Weight matrix.
            param (int): Parameter index.

        Returns:
            np.ndarray: Updated Gram matrix.
        """
        if self.method[param] == "ols":
            return update_inverted_gram(gram, X=x, w=w, forget=self.forget[param])
        if self.method[param] == "lasso":
            return update_gram(gram, X=x, w=w, forget=self.forget[param])

    def update_y_gram(
        self, gram: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray, param: int
    ) -> np.ndarray:
        """
        Update the Gram matrix for the response variable.

        Args:
            gram (np.ndarray): Current Gram matrix for the response variable.
            x (np.ndarray): Covariate matrix.
            y (np.ndarray): Response variable.
            w (np.ndarray): Weight matrix.
            param (int): Parameter index.

        Returns:
            np.ndarray: Updated Gram matrix for the response variable.
        """
        return update_y_gram(gram, X=x, y=y, w=w, forget=self.forget[param])

    @staticmethod
    def _make_intercept(n_observations: int) -> np.ndarray:
        """Make the intercept series as N x 1 array.

        Args:
            y (np.ndarray): Response variable $Y$

        Returns:
            np.ndarray: Intercept array.
        """
        return np.ones((n_observations, 1))

    @staticmethod
    def _add_lags(
        y: np.ndarray, x: np.ndarray, lags: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add lagged variables to the response and covariate matrices.

        Args:
            y (np.ndarray): Response variable.
            x (np.ndarray): Covariate matrix.
            lags (Union[int, np.ndarray]): Number of lags to add.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the updated response and covariate matrices.
        """
        if lags == 0:
            return y, x

        if isinstance(lags, int):
            lags = np.arange(1, lags + 1, dtype=int)

        max_lag = np.max(lags)
        lagged = np.stack([np.roll(y, i) for i in lags], axis=1)[max_lag:, :]
        new_x = np.hstack((x, lagged))[max_lag:, :]
        new_y = y[max_lag:]
        return new_y, new_x

    def _make_matrix_or_intercept(
        self, n_observations: int, x: np.ndarray, add_intercept: bool, param: int
    ):
        """
        Make the covariate matrix or an intercept array based on the input.

        Args:
            y (np.ndarray): Response variable.
            x (np.ndarray): Covariate matrix.
            add_intercept (bool): Flag indicating whether to add an intercept.
            param (int): Parameter index.

        Returns:
            np.ndarray: Matrix or intercept array.
        """
        if x is None:
            return self._make_intercept(n_observations=n_observations)
        elif add_intercept[param]:
            return self._add_intercept(x=x, param=param)
        else:
            return x
