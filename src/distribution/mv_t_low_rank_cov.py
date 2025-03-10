from itertools import product
from typing import Dict

import autograd.numpy as anp
import numpy as np
import scipy.special as sp
from autograd import elementwise_grad
from rolch.base import Distribution


def batched_log_lilkelihood_t_low_rank(y, mu, mat_d, mat_v, dof):
    k = y.shape[1]
    precision = anp.linalg.inv(mat_d + mat_v @ anp.swapaxes(mat_v, -2, -1))

    A = np.squeeze(sp.gammaln((dof + k) / 2))
    B = np.squeeze(sp.gammaln(dof / 2))
    C = 1 / 2 * k * np.squeeze(np.log(np.pi * dof))
    Z = np.sum((y - mu) * (precision @ (y - mu)[..., None]).squeeze(), 1)

    part1 = A - B - C
    part2 = 1 / 2 * np.log(np.linalg.det(precision))
    part3 = np.squeeze((dof + k) / 2) * np.log((1 + np.squeeze((1 / dof)) * Z))
    return part1 + part2 - part3


def first_partial_mu(y, mu, mat_d, mat_v, dof, i):
    k = y.shape[1]
    precision = np.linalg.inv(mat_d + mat_v @ np.swapaxes(mat_v, -2, -1))
    Z = anp.sum((y - mu) * (precision @ (y - mu)[..., None]).squeeze(), 1)
    part1 = (k + dof) / (2 * (Z + dof))
    part2 = 2 * np.sum(precision[:, i, :] * (y - mu), axis=1)
    return part1 * part2


def second_partial_mu(y, mu, mat_d, mat_v, dof, i):
    k = y.shape[1]

    precision = np.linalg.inv(mat_d + mat_v @ np.swapaxes(mat_v, -2, -1))
    Z = anp.sum((y - mu) * (precision @ (y - mu)[..., None]).squeeze(), 1)
    deriv_1 = 2 * np.sum(precision[:, i, :] * (y - mu), axis=1)
    deriv_2 = 2 * precision[:, i, i]

    part1 = k + dof
    part2 = (Z + dof) * deriv_2 - deriv_1**2
    part3 = 2 * (Z + dof) ** 2

    return -(part1 * part2) / part3


def partial1_D_element(y, mat_mu, mat_d, mat_v, dof, i):
    k = y.shape[1]
    cov = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    precision = np.linalg.inv(cov)
    Z = anp.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    # Derivative of the log(det(D + VV^T))
    part_1 = -0.5 * precision[:, i, i]

    # Derivative of the last term
    part2 = np.squeeze(k + dof) / (2 * (Z + dof.squeeze()))
    part3 = -np.squeeze(precision @ (y - mat_mu)[..., None])[:, i] ** 2
    return part_1 - part2 * part3


def partial2_D_element(y, mat_mu, mat_d, mat_v, dof, i):
    k = y.shape[1]
    cov = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    precision = np.linalg.inv(cov)
    Z = anp.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    # Derivative for the log(det())
    part1 = 0.5 * precision[:, i, i] ** 2

    # Derivative for the last term
    deriv_lambda_1 = -np.squeeze(precision @ (y - mat_mu)[..., None])[:, i] ** 2
    deriv_lambda_2 = (
        2
        * np.squeeze(precision @ (y - mat_mu)[..., None])[:, i] ** 2
        * precision[:, i, i]
    )

    part2 = k + dof.squeeze()
    part3 = (Z + dof.squeeze()) * deriv_lambda_2 - deriv_lambda_1**2
    part4 = 2 * (Z + dof.squeeze()) ** 2
    return part1 - (part2 * part3) / part4


def partial1_V_element(y, mat_mu, mat_d, mat_v, dof, i, j):
    # Would be nice to calculate only the necessary rows of Omega in the future maybe!
    # For part 2
    # zzT @ V
    # zzT[:, i, :] @ mat_v[:, :, j]
    # select the correct row of zzT before
    # sum(z * z[:, i], axis=-1)
    k = y.shape[1]

    cov = mat_d + mat_v @ anp.swapaxes(mat_v, -2, -1)
    precision = anp.linalg.inv(cov)
    Z = anp.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    # Deriviative for the log(det())
    part1 = anp.sum(precision[:, i, :] * mat_v[:, :, j], axis=1)

    # Derivative for the second part
    deriv = (
        -2
        * (
            precision
            @ (y - mat_mu)[..., None]
            @ (y - mat_mu)[:, None, :]
            @ precision
            @ mat_v
        )[:, i, j]
    )
    # Factor
    part2 = anp.squeeze(k + dof)
    part3 = 2 * (Z + dof.squeeze())
    return -part1 - (part2 / part3) * deriv


def partial2_V_element(y, mat_mu, mat_d, mat_v, dof, i, j):
    grad = elementwise_grad(partial1_V_element, argnum=3)
    return grad(y, mat_mu, mat_d, mat_v, dof, i, j)[:, i, j]


def first_partial_dof(y, mat_mu, mat_d, mat_v, dof):
    k = y.shape[1]

    precision = np.linalg.inv(mat_d + mat_v @ np.swapaxes(mat_v, -2, -1))
    Z = anp.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    part1 = -(-dof * sp.digamma((k + dof) / 2) + k + dof * sp.digamma(dof / 2)) / (
        2 * dof
    )
    part2 = (Z * (k + dof)) / (2 * dof * (dof + Z)) - 1 / 2 * np.log((dof + Z) / dof)
    return part1 + part2


def second_partial_dof(y, mat_mu, mat_d, mat_v, dof):
    k = y.shape[1]

    precision = np.linalg.inv(mat_d + mat_v @ np.swapaxes(mat_v, -2, -1))
    Z = anp.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    part1 = (
        1
        / 4
        * (
            (2 * k) / (dof**2)
            + sp.polygamma(1, (dof + k) / 2)
            - sp.polygamma(1, dof / 2)
        )
    )
    part2 = (Z * (dof * Z - k * (2 * dof + Z))) / (2 * dof**2 * (dof + Z) ** 2)
    return part1 + part2


class MultivariateStudentTLowRank(Distribution):

    # The cholesky decomposition of
    # COV = L @ L.T
    # PRC = (L^-1).T @ (L^-1)
    # The cross derivatives are defined on (L^-1).T
    # The transpose is important!!

    def __init__(
        self, loc_link, scale_link_1, scale_link_2, tail_link, rank, dof_guesstimate=1e6
    ):
        self.loc_link = loc_link
        self.scale_link_1 = scale_link_1
        self.scale_link_2 = scale_link_2
        self.tail_link = tail_link
        self.rank = rank
        self.dof_guesstimate = dof_guesstimate

        self.is_multivariate = True
        self.n_params = 4
        self.links = {
            0: self.loc_link,
            1: self.scale_link_1,
            2: self.scale_link_2,
            3: self.tail_link,
        }

        # Private API
        self._param_structure = {
            0: "matrix",
            1: "square_matrix",
            2: "matrix",
            3: "vector",
        }
        # We should rename this.
        self._adr_lower_diag = {0: False, 1: False, 2: False, 3: False}
        self._regularization = "low_rank"  # or adr
        self._regularization_allowed = {0: False, 1: False, 2: True, 3: False}

        self._scoring = "fisher"
        self._check_links()

    def _check_links(self):
        for p in range(self.n_params):
            if self.param_structure[p] not in self.links[p]._valid_structures:
                raise ValueError(
                    f"Link function does not match parameter structure for parameter {p}. \n"
                    f"Parameter structure is {self.param_structure[p]}. \n"
                    f"Link function supports {self.links[p]._valid_structures}"
                )

    def fitted_elements(self, dim: int):
        return {0: dim, 1: dim, 2: dim * self.rank, 3: 1}

    @property
    def param_structure(self):
        return self._param_structure

    def index_flat_to_cube(self, k: int, d: int, param: int):
        if (param == 0) | (param == 3):
            return k
        if param == 1:
            i, j = np.diag_indices(d)
            return i[k], j[k]
        if param == 2:
            idx = [(j, i) for i, j in product(range(self.rank), range(d))]
            return idx[k][0], idx[k][1]

    def set_theta_element(
        self, theta: Dict, value: np.ndarray, param: int, k: int
    ) -> Dict:
        """Sets an element of theta for parameter param and place k.

        !!! Note
            This will mutate `theta`!

        Args:
            theta (Dict): Current fitted $\theta$
            value (np.ndarray): Value to set
            param (int): Distribution parameter
            k (int): Flat element index $k$

        Returns:
            Dict: Theta where element (param, k) is set to value.
        """
        if (param == 0) | (param == 3):
            theta[param][:, k] = value
        if (param == 1) | (param == 2):
            d = theta[0].shape[1]
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            theta[param][:, i, j] = value
        return theta

    def theta_to_params(self, theta):
        loc = theta[0]
        mat_d = theta[1]
        mat_v = theta[2]
        mat_dof = theta[3].squeeze()
        return loc, mat_d, mat_v, mat_dof

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = False
    ):
        mu, mat_d, mat_v, dof = self.theta_to_params(theta)
        d = y.shape[1]
        if param == 0:
            deriv = first_partial_mu(y, mu, mat_d, mat_v, dof, i=k)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = partial1_D_element(y, mu, mat_d, mat_v, dof, i=i)
        if param == 2:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = partial1_V_element(y, mu, mat_d, mat_v, dof, i=i, j=j)
        if param == 3:
            deriv = first_partial_dof(y, mu, mat_d, mat_v, dof)
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = True
    ):
        mu, mat_d, mat_v, dof = self.theta_to_params(theta)
        d = y.shape[1]
        if param == 0:
            deriv = second_partial_mu(y, mu, mat_d, mat_v, dof, i=k)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = partial2_D_element(y, mu, mat_d, mat_v, dof, i=i)
        if param == 2:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = partial2_V_element(y, mu, mat_d, mat_v, dof, i=i, j=j)
        if param == 3:
            deriv = second_partial_dof(y, mu, mat_d, mat_v, dof)

        if clip:
            deriv = np.clip(deriv, -np.inf, -1e-10)

        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def link_function(self, y, param=0):
        return self.links[param].link(y)

    def link_inverse(self, y, param=0):
        return self.links[param].inverse(y)

    def link_function_derivative(self, y, param=0):
        return self.links[param].link_derivative(y)

    def link_inverse_derivative(self, y, param=0):
        return self.links[param].inverse_derivative(y)

    def link_function_second_derivative(self, y, param=0):
        return self.links[param].link_second_derivative(y)

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_link(y, i=i, j=j)

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2) | (param == 3):
            return self.links[param].inverse(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_inverse(y, i=i, j=j)

    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2) | (param == 3):
            return self.links[param].link_derivative(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_link_derivative(y, i=i, j=j)

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2) | (param == 3):
            return self.links[param].link_second_derivative(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_link_second_derivative(y, i=i, j=j)

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2) | (param == 3):
            return self.links[param].inverse_derivative(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_inverse_derivative(y, i=i, j=j)

    def initial_values(self, y: np.ndarray, param: int = 0):
        M = y.shape[0]
        if param == 0:
            return np.tile(np.mean(y, axis=0), (M, 1))
        if param == 1:
            mat_d = np.diag(np.var(y, 0))
            return np.tile(mat_d, (M, 1, 1))
        if param == 2:
            M = y.shape[0]
            omega = np.cov(y, rowvar=False)
            mat_d = np.diag(np.var(y, 0))
            eig = np.linalg.eig(omega - mat_d)
            largest_ev = np.argsort(eig.eigenvalues)[-self.rank :][::-1]
            mat_v = eig.eigenvectors[:, largest_ev]
            return np.tile(mat_v, (M, 1, 1))
        if param == 3:
            return np.full((M, 1), self.dof_guesstimate)

    def cube_to_flat(self, x: np.ndarray, param: int):
        if (param == 0) | (param == 3):
            return x
        if param == 1:
            return np.copy(x.diagonal(axis1=1, axis2=2))
        if param == 2:
            return x.swapaxes(-1, -2).reshape((x.shape[0], np.prod(x.shape[1:])))

    def flat_to_cube(self, x: np.ndarray, param: int):
        if (param == 0) | (param == 3):
            return x
        if param == 1:
            d = x.shape[1]
            out = np.zeros((x.shape[0], d, d))
            out[:, np.arange(d), np.arange(d)] = x
            return out
        if param == 2:
            d = int(x.shape[1] // self.rank)
            return x.reshape((x.shape[0], self.rank, d)).transpose(0, 2, 1)

    def log_likelihood(self, y: np.ndarray, theta: Dict[int, np.ndarray]):
        loc, mat_d, mat_v, dof = self.theta_to_params(theta)
        return batched_log_lilkelihood_t_low_rank(
            y, loc, mat_d=mat_d, mat_v=mat_v, dof=dof
        )

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        raise NotImplementedError("Not implemented")
