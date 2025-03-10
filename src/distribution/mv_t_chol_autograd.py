# Author Simon Hirsch
# MIT Licence
from typing import Dict

import autograd.numpy as anp
import autograd.scipy.special as asp
import numpy as np
from autograd import elementwise_grad
from rolch.base import Distribution


class MultivariateStudentTInverseCholeskyAutograd(Distribution):

    # The cholesky decomposition of
    # COV = L @ L.T
    # PRC = (L^-1).T @ (L^-1)
    # The cross derivatives are defined on (L^-1).T
    # The transpose is important!!

    def __init__(self, loc_link, scale_link, tail_link):
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.tail_link = tail_link

        self.is_multivariate = True
        self.n_params = 3
        self.links = {0: self.loc_link, 1: self.scale_link, 2: self.tail_link}

        # Private API
        self._param_structure = {
            0: "matrix",
            1: "square_matrix",
            2: "vector",
        }
        self._adr_lower_diag = {0: False, 1: True, 2: False}
        self._regularization_allowed = {0: False, 1: True, 2: False}
        self._regularization = "adr"
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

    @staticmethod
    def fitted_elements(dim: int):
        return {0: dim, 1: int(dim * (dim + 1) // 2), 2: 1}

    @property
    def param_structure(self):
        return self._param_structure

    @staticmethod
    def index_flat_to_cube(k: int, d: int, param: int):
        if (param == 0) | (param == 2):
            return k
        if param == 1:
            i, j = np.triu_indices(d)
            return i[k], j[k]

    @staticmethod
    def set_theta_element(theta: Dict, value: np.ndarray, param: int, k: int) -> Dict:
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
        if (param == 0) | (param == 2):
            theta[param][:, k] = value
        if param == 1:
            d = theta[0].shape[1]
            i, j = np.triu_indices(d)
            theta[param][:, i[k], j[k]] = value
        return theta

    def theta_to_params(self, theta):
        loc = theta[0]
        inv_tr_chol = theta[1]
        dof = theta[2]
        return loc, inv_tr_chol, dof

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
        mu, chol, dof = self.theta_to_params(theta)
        if param == 0:
            deriv = elem_grad_mu_1st_idx(y, mu, chol, dof, i=k, j=0)
        if param == 1:
            ii, jj = np.triu_indices(y.shape[1])
            deriv = elem_grad_cov_1st_idx(y, mu, chol, dof, i=ii[k], j=jj[k])
        if param == 2:
            deriv = elem_grad_dof_1st_idx(y, mu, chol, dof, i=k, j=0).squeeze()
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = True
    ):
        mu, chol, dof = self.theta_to_params(theta)
        if param == 0:
            deriv = elem_grad_mu_2nd_idx(y, mu, chol, dof, i=k, j=0)
        if param == 1:
            ii, jj = np.triu_indices(y.shape[1])
            deriv = elem_grad_cov_2nd_idx(y, mu, chol, dof, i=ii[k], j=jj[k])
        if param == 2:
            deriv = elem_grad_dof_2nd_idx(y, mu, chol, dof, i=k, j=0).squeeze()

        if clip:
            deriv = np.clip(deriv, -np.inf, -1e-5)

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

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_link(y, i=i[k], j=j[k])

    def element_link_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link_derivative(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_link_derivative(y, i=i[k], j=j[k])

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].inverse(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_inverse(y, i=i[k], j=j[k])

    def element_link_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link_derivative(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_link_derivative(y, i=i[k], j=j[k])

    def initial_values(self, y, param=0):
        M = y.shape[0]
        if param == 0:
            return np.tile(np.mean(y, axis=0), (M, 1))
        if param == 1:
            chol = np.linalg.inv(np.linalg.cholesky(np.cov(y, rowvar=False))).T
            return np.tile(chol, (M, 1, 1))
        if param == 2:
            return np.full((y.shape[0], 1), 10)

    def cube_to_flat(self, x: np.ndarray, param: int):
        if (param == 0) | (param == 2):
            return x
        if param == 1:
            d = x.shape[1]
            i = np.triu_indices(d)
            out = x[:, i[0], i[1]]
            return out

    def flat_to_cube(self, x: np.ndarray, param: int):
        if (param == 0) | (param == 2):
            return x
        if param == 1:
            n, k = x.shape
            # The following conversion holds for upper diagonal matrices
            # We INCLUDE the diagonal!!
            # (D + 1) * D // 2 = k
            # (D + 1) * D = 2 * k
            # D^2 + D = 2 * k
            # ... Wolfram Alpha ...
            # D = 0.5 * (sqrt(8k + 1) - 1)
            d = int(1 / 2 * (np.sqrt(8 * k + 1) - 1))
            i = np.triu_indices(d)
            out = np.zeros((n, d, d))
            out[:, i[0], i[1]] = x
            return out

    def log_likelihood(self, y: np.ndarray, theta: Dict[int, np.ndarray]):
        loc, inv_tr_chol, dof = self.theta_to_params(theta)
        return batched_log_likelihood(y, loc, inv_tr_chol, dof)

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        raise NotImplementedError("Not implemented")


##########################################################
### numba JIT-compiled functions for the derivatives #####
##########################################################


def batched_log_likelihood(y, mu, chol, dof):
    D = y.shape[1]
    A = anp.squeeze(asp.gamma((dof + D) / 2))
    B = anp.squeeze(asp.gamma(dof / 2) * dof ** (D / 2) * np.pi ** (D / 2))
    Z = anp.sum(anp.einsum("nd, nkd -> nk", (y - mu), chol) ** 2, axis=1)
    return (
        anp.log(A / B)
        + anp.log(anp.linalg.det(chol))
        + anp.log((1 + anp.squeeze((1 / dof)) * Z) ** -anp.squeeze((dof + D) / 2))
    )


jac1 = elementwise_grad(batched_log_likelihood, argnum=1)
jac2 = elementwise_grad(batched_log_likelihood, argnum=2)
jac3 = elementwise_grad(batched_log_likelihood, argnum=3)


def elem_grad_mu_1st_idx(y, mu, chol, dof, i, j):
    return jac1(y, mu, chol, dof)[:, i]


def elem_grad_cov_1st_idx(y, mu, chol, dof, i, j):
    return jac2(y, mu, chol, dof)[:, i, j]


def elem_grad_dof_1st_idx(y, mu, chol, dof, i, j):
    return jac3(y, mu, chol, dof)


hess1 = elementwise_grad(elem_grad_mu_1st_idx, argnum=1)
hess2 = elementwise_grad(elem_grad_cov_1st_idx, argnum=2)
hess3 = elementwise_grad(elem_grad_dof_1st_idx, argnum=3)


def elem_grad_mu_2nd_idx(y, mu, chol, dof, i, j):
    return hess1(y, mu, chol, dof, i, j)[:, i]


def elem_grad_cov_2nd_idx(y, mu, chol, dof, i, j):
    return hess2(y, mu, chol, dof, i, j)[:, i, j]


def elem_grad_dof_2nd_idx(y, mu, chol, dof, i, j):
    return hess3(y, mu, chol, dof, i, j)
