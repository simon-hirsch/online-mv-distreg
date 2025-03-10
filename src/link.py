from typing import Tuple

import numba as nb
import numpy as np
from rolch.base import LinkFunction
from rolch.link import EXP_UPPER_BOUND, LOG_LOWER_BOUND, LogIdentLink


@nb.vectorize(["float32(float32, float32)", "float64(float64, float64)"])
def robust_sigmoid(value, k):
    if value / k > 50:
        return 1
    elif value / k < -50:
        return 0
    else:
        return 1 / (1 + np.exp(-k * (value - 1)))


@nb.vectorize(["float32(float32, float32)", "float64(float64, float64)"])
def zero_safe_division(a, b):
    """Return the max float value for the current precision at the a / b if
    b becomes 0. Returns

    Args:
        a (float): Nominator.
        b (float): Denominator.

    Returns:
        float: Result of the division a / b.
    """
    if np.isclose(b, 0):
        return 0
    elif np.isclose(a, 0):
        return 0
    else:
        return a / b


class LogShiftValueLink(LinkFunction):
    """
    The Log-Link function shifted to a value \(v\).

    This link function is defined as \(g(x) = \log(x - v)\). It can be used
    to ensure that certain distribution paramters don't fall below lower
    bounds, e.g. ensuring that the degrees of freedom of a Student's t distribtuion
    don't fall below 2, hence ensuring that the variance exists.
    """

    def __init__(self, value: float):
        self.value = value

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.log(x - self.value + LOG_LOWER_BOUND)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.value + np.fmax(
            np.exp(np.fmin(x, EXP_UPPER_BOUND)), LOG_LOWER_BOUND
        )

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.fmax(np.exp(np.fmin(x, EXP_UPPER_BOUND)), LOG_LOWER_BOUND)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (x - self.value + 0.1)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1 / (x - self.value + 0.1) ** 2


class LogShiftTwoLink(LogShiftValueLink):
    """
    The Log-Link function shifted to 2.

    This link function is defined as \(g(x) = \log(x - 2)\). It can be used
    to ensure that certain distribution paramters don't fall below lower
    bounds, e.g. ensuring that the degrees of freedom of a Student's t distribtuion
    don't fall below 2, hence ensuring that the variance exists.
    """

    def __init__(self):
        super().__init__(2)
        pass


class InverseSoftPlusLink(LinkFunction):
    """
    The softplus is defined as $$
        \operatorname{SoftPlus(x)} = \log(1 + \exp(x))
    $$ and hence the inverse is defined as $$
        \log(\exp(x) - 1)
    $$ which can be used as link function for the parameters on the
    positive real line. The behavior of the inverse softplus is more
    graceful on large values as it avoids the exp of the log-link and
    converges towards a linear behaviour.

    The softplus is the smooth approximation of $\max(x, 0)$.
    """

    def __init__(self):
        pass

    def link(self, x: np.ndarray):
        return np.log(np.clip(np.exp(x) - 1, 1e-10, np.inf))

    def inverse(self, x: np.ndarray):
        return np.log(1 + np.exp(x))

    def link_derivative(self, x: np.ndarray):
        return 1 / (np.exp(x) - 1) + 1

    def link_second_derivative(self, x: np.ndarray):
        # return 1 / (2 - 2 * np.cosh(x)) # equivalent
        return -1 / (np.exp(x) - 1) - 1 / (np.exp(x) - 1) ** 2

    def inverse_derivative(self, x: np.ndarray):
        return 1 / (np.exp(-x) + 1)


class InverseSoftPlusShiftValueLink(LinkFunction):
    """
    The Inverse SoftPlus function shifted to a value \(v\).
    """

    def __init__(self, value: float):
        self.value = value

    def link(self, x: np.ndarray):
        return np.log(np.exp(x - self.value) - 1)

    def inverse(self, x: np.ndarray):
        return np.log(1 + np.exp(x)) + self.value

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, (np.exp(-x) + 1))

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, (1 - np.exp(self.value - x)))

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, (2 - 2 * np.cosh(self.value - x)))


class InverseSoftPlusShiftTwoLink(InverseSoftPlusShiftValueLink):
    """
    The Inverse SoftPlus function shifted to 2.
    """

    def __init__(self):
        super().__init__(2)
        pass


class TwiceDifferentiableLogIdentLink(LogIdentLink):

    def __init__(self, k=1):
        self.k = k

    def link(self, x: np.ndarray):
        return np.where(
            x <= 1,
            np.log(x),
            (1 - robust_sigmoid(x, self.k)) * np.log(x)
            + robust_sigmoid(x, self.k) * (x - 1),
        )

    def link_derivative(self, x):
        sigm = robust_sigmoid(x, self.k)
        deriv = (
            (1 - sigm) * (1 / x)
            + self.k * sigm * (1 - sigm) * (np.log(x) - (x - 1))
            + sigm
        )
        return deriv

    def link_second_derivative(self, x):
        sigm = robust_sigmoid(x, self.k)
        sigm_d1 = self.k * sigm * (1 - sigm)
        sigm_d2 = self.k**2 * sigm * (1 - sigm) * (1 - 2 * sigm)
        deriv = (
            -sigm_d1 / x
            - (1 - sigm) / x**2
            + sigm_d2 * (np.log(x) - (x - 1))
            + sigm_d1 * (1 / x)
        )
        return deriv


class LogIdentShiftValueLink(LinkFunction):

    def __init__(self, value):
        self.value = value

    def link(self, x: np.ndarray):
        return np.where(
            x < (1 + self.value),
            np.log(
                np.fmax(LOG_LOWER_BOUND, x - self.value)
            ),  # Ensure that everything is robust
            x - 1 - self.value,
        )

    def inverse(self, x):
        return np.where(
            x <= 0,
            self.value + np.exp(np.fmin(x, EXP_UPPER_BOUND)),
            x + 1 + self.value,
        )

    def link_derivative(self, x):
        # return np.where(x < (1 + self.value), 1 / (x - self.value), 1)
        raise ValueError("Not  continuous differentiable.")

    def link_second_derivative(self, x):
        raise ValueError("Not twice continuous differentiable.")

    def inverse_derivative(self, x):
        return np.where(x <= 0, np.exp(np.fmin(x - 1, EXP_UPPER_BOUND)), 1)


class MatrixDiagLink(LinkFunction):
    """
    Wraps a link functions to be applied only on the diagonal of a square matrix.
    """

    def __init__(self, diag_link: LinkFunction, other_val=0):
        self.diag_link = diag_link
        self.other_val = other_val
        self._valid_structures = ["square_matrix"]

    def _make_indices(self, x: np.ndarray) -> Tuple:
        d = x.shape[1]
        i = np.diag_indices(d)
        return i

    def link(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.link(x[:, i[0], i[1]])
        return out

    def inverse(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.inverse(x[:, i[0], i[1]])
        return out

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.link_derivative(x[:, i[0], i[1]])
        return out

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.link_second_derivative(x[:, i[0], i[1]])
        return out

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.inverse_derivative(x[:, i[0], i[1]])
        return out

    def element_link(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")

    def element_link_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link_derivative(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")

    def element_link_second_derivative(
        self, x: np.ndarray, i: int, j: int
    ) -> np.ndarray:
        if i == j:
            return self.diag_link.link_second_derivative(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")

    def element_inverse_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse_derivative(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")

    def element_inverse(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")


class MatrixDiagTriuLink(LinkFunction):
    """
    Wraps two link functions to be applied on the diagonal and the upper diagonal of a square matrix.
    """

    def __init__(self, diag_link: LinkFunction, triu_link: LinkFunction):
        self.diag_link = diag_link
        self.triu_link = triu_link

        # For checking
        self._valid_structures = ["square_matrix"]

    def _make_indices(self, x: np.ndarray) -> Tuple:
        d = x.shape[1]
        i = np.diag_indices(d)
        j = np.triu_indices(d, k=1)
        return i, j

    def link(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.link(x[:, j[0], j[1]])
        return out

    def inverse(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.inverse(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.inverse(x[:, j[0], j[1]])
        return out

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.link_derivative(x[:, j[0], j[1]])
        return out

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link_second_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.link_second_derivative(x[:, j[0], j[1]])
        return out

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.inverse_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.inverse_derivative(x[:, j[0], j[1]])
        return out

    def element_link(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link(x)
        else:
            return self.triu_link.link(x)

    def element_link_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link_derivative(x)
        else:
            return self.triu_link.link_derivative(x)

    def element_link_second_derivative(
        self, x: np.ndarray, i: int, j: int
    ) -> np.ndarray:
        if i == j:
            return self.diag_link.link_second_derivative(x)
        else:
            return self.triu_link.link_second_derivative(x)

    def element_inverse_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.element_inverse_derivative(x)
        else:
            return self.triu_link.element_inverse_derivative(x)

    def element_inverse(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse(x)
        else:
            return self.triu_link.inverse(x)


class MatrixDiagTrilLink(LinkFunction):

    def __init__(self, diag_link: LinkFunction, tril_link: LinkFunction):
        self.diag_link = diag_link
        self.tril_link = tril_link

        # For checking
        self._valid_structures = ["square_matrix"]

    def _make_indices(self, x: np.ndarray) -> Tuple:
        d = x.shape[1]
        i = np.diag_indices(d)
        j = np.tril_indices(d, k=-1)
        return i, j

    def link(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.link(x[:, j[0], j[1]])
        return out

    def inverse(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.inverse(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.inverse(x[:, j[0], j[1]])
        return out

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.link_derivative(x[:, j[0], j[1]])
        return out

    def inverse_derivative(self, x):
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.inverse_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.inverse_derivative(x[:, j[0], j[1]])
        return out

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link_second_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.link_second_derivative(x[:, j[0], j[1]])
        return out

    def element_link(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link(x)
        else:
            return self.tril_link.link(x)

    def element_link_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link_derivative(x)
        else:
            return self.tril_link.link_derivative(x)

    def element_link_second_derivative(
        self, x: np.ndarray, i: int, j: int
    ) -> np.ndarray:
        if i == j:
            return self.diag_link.link_second_derivative(x)
        else:
            return self.tril_link.link_second_derivative(x)

    def element_inverse_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse_derivative(x)
        else:
            return self.tril_link.inverse_derivative(x)

    def element_inverse(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse(x)
        else:
            return self.tril_link.inverse(x)
