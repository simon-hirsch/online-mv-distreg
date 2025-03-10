import copy
import time
import warnings
from typing import Dict, List, Literal, Optional, Union

import numba as nb
import numpy as np
from rolch.base import Distribution
from rolch.coordinate_descent import online_coordinate_descent_path
from rolch.utils import calculate_effective_training_length, handle_param_dict

from ..base import Estimator
from ..estimator.utils import (
    get_adr_regularization_distance,
    get_max_lambda,
    indices_along_diagonal,
    information_criteria_likelihood,
    make_model_array,
)
from ..scaler import MultivariateOnlineScaler

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class MultivariateOnlineGamlss(Estimator):

    def __init__(
        self,
        distribution: Distribution,
        equation: Dict,
        scale_inputs: bool = True,
        verbose: int = 1,
        method: Dict[int, Literal["ols", "lasso"]] = {0: "lasso", 1: "lasso"},
    ):
        self.distribution = distribution

        # For simplicity
        super().__init__(method=method)
        self.forget = {0: 0, 1: 0}
        self.learning_rate = 0
        self.equation = equation
        self.fit_intercept = {0: True, 1: True}
        self.iteration_along_diagonal = False

        # TODO: This has currently no effect.
        self.regularize_intercept = {0: False, 1: True}
        self.adr_regularization = {0: 0, 1: 0}

        self.approx_fast_model_selection = True

        # Scaler
        self.scale_inputs = scale_inputs
        self.scaler = MultivariateOnlineScaler(
            forget=self.learning_rate, do_scale=self.scale_inputs
        )

        # For LASSO
        self.lambda_n = 100
        self.lambda_eps = 1e-4
        self.lambda_grid = {}
        self.ic = "aic"

        # Iterations and Blabla
        self.max_iterations_outer = 10
        self.max_iterations_inner = 10
        self.rel_tol_inner = 1e-3
        self.abs_tol_inner = 1e-3
        self.rel_tol_outer = 1e-3
        self.abs_tol_outer = 1e-3

        # For pretty printing.
        # TODO: This can be moved to top classes
        self._verbose_prefix = f"[{self.__class__.__name__}]"
        self._verbose_end = {1: "\n", 2: "\r", 3: "\r"}
        self.verbose = verbose

    # Same for Univariate and Multivariate
    def handle_default(self, value, default, name):
        handle_param_dict(
            self,
            param=value,
            default=default,
            name=name,
            n_params=self.distribution.n_params,
        )

    def make_iteration_indices(self, param: int):

        if (
            self.distribution._param_structure == "square_matrix"
        ) & self.iteration_along_diagonal:
            index = np.arange(indices_along_diagonal(self.D[param]))
        else:
            index = np.arange(self.K[param])

        return index

    # Different UV - MV
    def make_initial_theta(self, y: np.ndarray):
        theta = {
            p: self.distribution.initial_values(y, p)
            for p in range(self.distribution.n_params)
        }
        # Handle the AD-R regularization
        self.prepare_adr_regularization()
        for p in range(self.distribution.n_params):
            if self.adr_regularization[p] > 0:
                mask = self.adr_distance[p] >= self.adr_regularization[p]
                regularized = self.distribution.cube_to_flat(theta[p], p)
                regularized[:, mask] = regularized[:, mask] = 0
                theta[p] = self.distribution.flat_to_cube(regularized, p)

        return theta

    # Only MV
    def is_element_adr_regularized(self, p: int, k: int):
        if self.adr_regularization[p] == 0:
            return False
        else:
            return self.adr_distance[p][k] >= self.adr_regularization[p]

    # Only MV
    def prepare_adr_regularization(self) -> None:
        self.adr_distance = {}
        for p, structure in self.distribution._param_structure.items():
            if structure == "square_matrix":
                self.adr_distance[p] = get_adr_regularization_distance(
                    d=self.D, lower_diag=self.distribution._adr_lower_diag
                )
            elif self.adr_regularization[p] == 0:
                pass  # TODO: This is probably not so good style
            else:
                warnings.warn(
                    f"{self._verbose_prefix} "
                    f"You have specified AD-r Regularization for parameter {p}. "
                    f"AD-r regularization is only possible for square matrices. "
                    f"Parameter {p} has the shape {structure}. "
                    f"Skipping AD-r regularization for this parameter."
                )
                self.adr_distance[p] = None
                self.adr_regularization[p] = 0

    # Different UV-MV
    def get_number_of_covariates(self, X: np.ndarray):
        J = {}
        for p in range(self.distribution.n_params):
            J[p] = {}
            for k in range(self.K[p]):
                if isinstance(self.equation[p][k], str):
                    if self.equation[p][k] == "all":
                        J[p][k] = X.shape[1] + int(self.fit_intercept[p])
                    if self.equation[p][k] == "intercept":
                        J[p][k] = 1
                elif isinstance(self.equation[p][k], np.ndarray) or isinstance(
                    self.equation[p][k], list
                ):
                    J[p][k] = len(self.equation[p][k]) + int(self.fit_intercept[p])
                else:
                    raise ValueError("Something unexpected happened")
        return J

    # Different UV-MV
    def validate_equation(self, equation):
        if equation is None:
            warnings.warn(
                f"[{self.__class__.__name__}] "
                "Equation is not specified. "
                "Per default, will estimate the first distribution parameter by all covariates found in X. "
                "All other distribution parameters will be estimated by an intercept."
            )
            equation = {
                p: (
                    {k: "all" for k in range(self.K[p])}
                    if p == 0
                    else {k: "intercept" for k in range(self.K[p])}
                )
                for p in range(self.distribution.n_params)
            }
        else:
            for p in range(self.distribution.n_params):
                # Check that all distribution parameters are in the equation.
                # If not, add intercept.
                if p not in equation.keys():
                    print(
                        f"{self.__class__.__name__}",
                        f"Distribution parameter {p} is not in equation.",
                        "All elements of the parameter will be estimated by an intercept.",
                    )
                    equation[p] = {k: "intercept" for k in range(self.K[p])}

                else:
                    for k in range(self.K[p]):
                        if k not in equation[p].keys():
                            print(
                                f"{self._verbose_prefix}",
                                f"Distribution parameter {p}, element {k} is not in equation.",
                                "Element of the parameter will be estimated by an intercept.",
                            )
                            equation[p][k] = "intercept"

                    if not (
                        isinstance(equation[p][k], np.ndarray)
                        or (equation[p][k] in ["all", "intercept"])
                    ):
                        if not (
                            isinstance(equation[p], list) and (HAS_PANDAS | HAS_POLARS)
                        ):
                            raise ValueError(
                                "The equation should contain of either: \n"
                                " - a numpy array of dtype int, \n"
                                " - a list of string column names \n"
                                " - or the strings 'all' or 'intercept' \n"
                                f"you have passed {equation[p]} for the distribution parameter {p}."
                            )

        return equation

    # Different UV - MV
    def fit(self, X, y):

        # Set fixed values
        self.n_observations = y.shape[0]
        self.n_effective_training = calculate_effective_training_length(
            forget=self.learning_rate, n_obs=self.n_observations
        )
        self.D = y.shape[1]
        self.K = self.distribution.fitted_elements(self.D)
        self.equation = self.validate_equation(self.equation)
        self.J = self.get_number_of_covariates(X)

        # Handle scaling
        self.scaler.fit(X, to_scale=np.arange(0, X.shape[1]))
        X_scaled = self.scaler.transform(x=X)

        # Some stuff
        self.is_regularized = {
            p: {k: np.repeat(True, self.J[p][k]) for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }

        self.iter_index = {
            p: self.make_iteration_indices(p) for p in range(self.distribution.n_params)
        }

        self.prepare_adr_regularization()
        theta = self._make_initial_theta(y)

        # Current information
        self.current_likelihood = np.sum(
            self.distribution.log_likelihood(y=y, theta=theta)
        )
        self.model_selection = {p: {} for p in range(self.distribution.n_params)}
        self.x_gram = {p: {} for p in range(self.distribution.n_params)}
        self.y_gram = {p: {} for p in range(self.distribution.n_params)}
        self.beta_path = {
            p: {k: np.zeros((self.lambda_n, self.J[p][k])) for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }
        self.beta = {
            p: {k: np.zeros((self.J[p][k])) for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }

        # Some information about the different iterations
        self.iteration_count = np.zeros(
            (self.max_iterations_outer, self.distribution.n_params), dtype=int
        )
        self.iteration_likelihood = np.zeros(
            (
                self.max_iterations_outer,
                self.max_iterations_inner,
                self.distribution.n_params,
            )
        )
        self.model_selection_ll = {
            p: np.zeros((self.lambda_n, self.K[p]))
            for p in range(self.distribution.n_params)
        }

        # TODO: Remove - We don't want to do this, for debugging only:
        self.X = X_scaled

        # Call the fit
        self._outer_fit(X=X_scaled, y=y, theta=theta)

        if self.verbose > 0:  # Level 1 Message
            print(
                self._verbose_prefix,
                "Finished fitting distribution parameters.",
                end=self._verbose_end[1],
            )

    def _outer_fit(self, X, y, theta):
        outer_start = time.time()
        for outer_iteration in range(self.max_iterations_outer):
            outer_it_start = time.time()
            # Breaking criteria
            global_likelihood = self.current_likelihood

            if outer_iteration == 0:
                global_old_likelihood = 0
            if outer_iteration > 0:
                if (
                    np.abs(global_likelihood - global_old_likelihood)
                    / np.abs(global_likelihood)
                    < self.rel_tol_outer
                ):
                    break
                if (
                    np.abs(global_likelihood - global_old_likelihood)
                    < self.abs_tol_outer
                ):
                    break
            global_old_likelihood = global_likelihood

            for p in range(self.distribution.n_params):
                if self.verbose >= 2:  #
                    print(
                        self._verbose_prefix,
                        f"Outer Iteration {outer_iteration}: Fitting Distribution parameter {p}",
                        end=self._verbose_end[2],
                    )
                theta = self._inner_fit(
                    X=X, y=y, theta=theta, outer_iteration=outer_iteration, p=p
                )

            outer_it_end = time.time()
            outer_it_time = outer_it_end - outer_it_start
            outer_it_avg = (outer_it_end - outer_start) / (outer_iteration + 1)
            outer_pred_time = outer_it_time * (
                self.max_iterations_outer - outer_iteration - 1
            )

            if self.verbose > 0:  # Level 1 message
                print(
                    self._verbose_prefix,
                    f"Last outer iteration {outer_iteration} took {round(outer_it_time, 1)} sec. "
                    f"Average outer iteration took {round(outer_it_avg, 1)} sec. "
                    f"Expected to be finished in max {round(outer_pred_time, 1)} sec. ",
                )
        self.theta = theta

    @staticmethod
    def count_nonzero_coef(beta):
        non_zero = 0
        for p, element_coef in beta.items():
            non_zero += int(np.sum([np.sum(c != 0) for c in element_coef.values()]))
        return non_zero

    def _make_initial_theta(self, y: np.ndarray):
        theta = {
            p: self.distribution.initial_values(y, p)
            for p in range(self.distribution.n_params)
        }
        # Handle AD-R Regularization
        for p in range(self.distribution.n_params):
            if self.adr_regularization[p] > 0:
                mask = self.adr_distance[p] >= self.adr_regularization[p]
                regularized = self.distribution.cube_to_flat(theta[p], p)
                regularized[:, mask] = regularized[:, mask] = 0
                theta[p] = self.distribution.flat_to_cube(regularized, p)

        return theta

    def _inner_fit(self, y, X, theta, outer_iteration, p):
        for inner_iteration in range(self.max_iterations_inner):
            if (inner_iteration == 0) and (outer_iteration == 0):
                old_likelihood = self.current_likelihood
            elif (inner_iteration == 0) and (outer_iteration > 0):
                old_likelihood = self.current_likelihood
            else:
                # Breaking criteria
                if (
                    np.abs(self.current_likelihood - old_likelihood)
                    / np.abs(self.current_likelihood)
                    < self.rel_tol_inner
                ):
                    break
                if (
                    np.abs(self.current_likelihood - old_likelihood)
                    < self.abs_tol_inner
                ):
                    break
                if ((outer_iteration > 0) | (inner_iteration > 1)) & (
                    self.current_likelihood < old_likelihood
                ):
                    warnings.warn("Likelihood is decreasing. Breaking.")
                    # Reset to values from the previous iteration
                    theta = prev_theta
                    self.x_gram[p] = prev_x_gram
                    self.y_gram[p] = prev_y_gram
                    # self.conditional_likelihood = prev_cond_ll
                    self.beta = prev_beta
                    self.beta_path = prev_beta_path
                    self.current_likelihood = old_likelihood
                    break

                old_likelihood = self.current_likelihood

            # Store previous iteration values:
            if (inner_iteration > 0) | (outer_iteration > 0):
                prev_theta = copy.copy(theta)
                prev_x_gram = copy.copy(self.x_gram[p])
                prev_y_gram = copy.copy(self.y_gram[p])
                # prev_cond_ll = copy.copy(self.conditional_likelihood)
                prev_model_selection = copy.copy(self.model_selection)
                prev_beta = copy.copy(self.beta)
                prev_beta_path = copy.copy(self.beta_path)

            # Iterate through all elements of the distribution parameter
            for k in self.iter_index[p]:
                if self.is_element_adr_regularized(p=p, k=k):
                    # Handle AD-R
                    self.beta[p][k] = np.zeros(self.J[p][k])
                    self.beta_path[p][k] = np.zeros((self.lambda_n, self.J[p][k]))
                else:
                    # TODO: This should be optimized at some future point.
                    # Still need to think about a good setup for the link
                    # and the flat-cube problem
                    eta = self.distribution.link_function(theta[p], p)
                    dr = 1 / self.distribution.cube_to_flat(
                        self.distribution.link_inverse_derivative(eta, param=p), param=p
                    )
                    eta = self.distribution.cube_to_flat(eta, param=p)
                    dr = dr[:, k]

                    dl1dp1 = self.distribution.element_score(
                        y, theta=theta, param=p, k=k
                    )
                    dl2dp2 = self.distribution.element_hessian(
                        y, theta=theta, param=p, k=k
                    )
                    if self.distribution._scoring == "fisher":
                        wt = -(dl2dp2 / (dr * dr))
                        wv = eta[:, k] + dl1dp1 / (dr * wt)
                    elif self.distribution._scoring == "newton_rapson":
                        wt = -dl2dp2
                        wv = eta[:, k] + dl1dp1 / wt
                    else:
                        raise ValueError("Unknown scoring method.")

                    # Base Estimator abstracts the method away
                    # This will create inverted grams for OLS and
                    # non-inverted grams for LASSO
                    x = make_model_array(
                        X=X,
                        eq=self.equation[p][k],
                        fit_intercept=self.fit_intercept[p],
                    )
                    self.x_gram[p][k] = self.make_gram(
                        x=x,
                        w=wt,
                        param=p,
                    )
                    self.y_gram[p][k] = self.make_y_gram(
                        x=x,
                        y=wv,
                        w=wt,
                        param=p,
                    )
                    if self.method[p] == "lasso":
                        lambda_max = get_max_lambda(
                            self.x_gram[p][k],
                            self.y_gram[p][k],
                            self.is_regularized[p][k],
                        )
                        lambda_path = np.geomspace(
                            lambda_max, lambda_max * self.lambda_eps, self.lambda_n
                        )

                        self.beta_path[p][k], _ = online_coordinate_descent_path(
                            x_gram=self.x_gram[p][k],
                            y_gram=self.y_gram[p][k].squeeze(axis=-1),
                            beta_path=self.beta_path[p][k],
                            lambda_path=lambda_path,
                            is_regularized=self.is_regularized[p][k],
                            beta_lower_bound=np.repeat(-np.inf, self.J[p][k]),
                            beta_upper_bound=np.repeat(np.inf, self.J[p][k]),
                            which_start_value="previous_lambda",
                            selection="cyclic",
                            tolerance=1e-4,
                            max_iterations=1000,
                        )
                        eta_elem = x @ self.beta_path[p][k].T
                        theta_elem = self.distribution.element_link_inverse(
                            eta_elem, param=p, k=k, d=self.D
                        )

                        # Model Selection should be factored into different method
                        # Model selection

                        if self.ic == "max":
                            opt_ic = self.lambda_n - 1
                        else:
                            theta_ll = copy.deepcopy(theta)

                            theta_elem_delta = np.diff(theta_elem, axis=1)
                            theta_ms = self.distribution.set_theta_element(
                                theta_ll, theta_elem[:, 0], param=p, k=k
                            )
                            approx_ll = self.distribution.log_likelihood(
                                y, theta_ll
                            ).sum()
                            approx_ll = np.repeat(approx_ll, 100)
                            for l_idx in range(1, self.lambda_n):
                                theta_ms = self.distribution.set_theta_element(
                                    theta_ll, theta_elem[:, l_idx], param=p, k=k
                                )
                                approx_ll[l_idx] = (
                                    approx_ll[l_idx - 1]
                                    + (
                                        self.distribution.element_dl1_dp1(
                                            y, theta_ms, param=p, k=k
                                        )
                                        * theta_elem_delta[:, l_idx - 1]
                                    ).sum()
                                )
                            # Count number of nonzero coefficients
                            # Subtract current beta if already fitted
                            # If in the first iteration, add intercept
                            # for all to-be-fitted parameters
                            # that are not AD-R regularized
                            nonzero = self.count_nonzero_coef(self.beta)
                            nonzero = -int(np.sum(self.beta[p][k] != 0))
                            if (outer_iteration == 0) & (inner_iteration == 0):
                                ## Add the "intercept" of the
                                ## next K - k parameters to the number of non-zero parameters
                                if self.adr_regularization[p] != 0:
                                    adr_zeroed = np.where(
                                        self.adr_distance[p]
                                        >= self.adr_regularization[p],
                                        0,
                                        1,
                                    )
                                else:
                                    adr_zeroed = np.ones(self.K[p])

                                nonzero = nonzero + np.sum(
                                    (np.ones(self.K[p]) * adr_zeroed)[k:]
                                )
                            ic = information_criteria_likelihood(
                                approx_ll, self.n_observations, nonzero, self.ic
                            )
                            self.model_selection[p][k] = {
                                "ll": approx_ll,
                                "non_zero": nonzero,
                                "ic": ic,
                            }
                            opt_ic = np.argmin(ic)

                        # select optimal beta and theta
                        self.beta[p][k] = self.beta_path[p][k][opt_ic, :]
                        theta = self.distribution.set_theta_element(
                            theta, theta_elem[:, opt_ic], param=p, k=k
                        )
                    elif self.method[p] == "ols":
                        self.beta[p][k] = (
                            self.x_gram[p][k] @ self.y_gram[p][k]
                        )  # ).squeeze()
                    else:
                        raise ValueError("Method not recognized")

                    eta[:, k] = np.squeeze(x @ self.beta[p][k])
                    theta[p] = self.distribution.link_inverse(
                        self.distribution.flat_to_cube(eta, param=p), param=p
                    )

                self.current_likelihood = self.distribution.log_likelihood(
                    y, theta=theta
                ).sum()

            if inner_iteration == (self.max_iterations_inner - 1):
                warnings.warn(
                    "Reached max inner iterations. Algorithm may or may not be converged."
                )
            self.iteration_count[outer_iteration, p] = inner_iteration
            self.iteration_likelihood[outer_iteration, inner_iteration, p] = (
                self.current_likelihood
            )
            if self.verbose >= 2:  # Level 2 Message
                print(
                    self._verbose_prefix,
                    f"Outer iteration: {outer_iteration}, inner iteration {inner_iteration}, parameter {p}:",
                    f"current likelihood: {self.current_likelihood},",
                    f"previous iteration likelihood {self.iteration_likelihood[outer_iteration, inner_iteration-1, p] if inner_iteration > 0 else self.current_likelihood}",
                    end=self._verbose_end[self.verbose],
                )

        return theta

    # Different UV - MV
    def predict(
        self,
        X: Optional[np.ndarray] = None,
    ) -> Dict[int, np.ndarray]:

        if X is None:
            X_scaled = np.ones((1, 1))
            N = 1
            print(self._verbose_prefix, "X is None. Prediction will have length 1.")
        else:
            X_scaled = self.scaler.transform(X)
            N = X.shape[0]
        out = {}

        for p in range(self.distribution.n_params):
            array = np.zeros((N, self.K[p]))
            for k in range(self.K[p]):
                array[:, k] = (
                    make_model_array(
                        X=X_scaled,
                        eq=self.equation[p][k],
                        fit_intercept=self.fit_intercept[p],
                    )
                    @ self.beta[p][k]
                ).squeeze()
            out[p] = self.distribution.flat_to_cube(array, p)
            out[p] = self.distribution.link_inverse(out[p], p)
        return out

    # Different UV - MV
    def update(self, X, y) -> None:
        self.n_observations += y.shape[0]
        self.n_effective_training = calculate_effective_training_length(
            forget=self.learning_rate, n_obs=self.n_observations
        )
        theta = self.predict(X)
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)

        self.x_gram_old = copy.deepcopy(self.x_gram)
        self.y_gram_old = copy.deepcopy(self.y_gram)
        self.model_selection_old = copy.deepcopy(self.model_selection)
        self.old_likelihood = self.current_likelihood + 0
        self.old_likelihood_discounted = (1 - self.learning_rate) * self.old_likelihood
        self.current_likelihood = self.old_likelihood_discounted + np.sum(
            self.distribution.log_likelihood(y=y, theta=theta)
        )
        self._outer_update(X=X_scaled, y=y, theta=theta)

    # Different UV - MV
    def _inner_update(self, X, y, theta, outer_iteration, p):

        for inner_iteration in range(self.max_iterations_inner):
            if (inner_iteration == 0) and (outer_iteration == 0):
                old_likelihood = self.current_likelihood
            elif (inner_iteration == 0) and (outer_iteration > 0):
                old_likelihood = self.current_likelihood
            else:
                # Breaking criteria
                if (
                    np.abs(self.current_likelihood - old_likelihood)
                    / np.abs(self.current_likelihood)
                    < self.rel_tol_inner
                ):
                    break
                if (
                    np.abs(self.current_likelihood - old_likelihood)
                    < self.abs_tol_inner
                ):
                    break
                old_likelihood = self.current_likelihood

            for k in self.iter_index[p]:
                # Handle AD-R Regularization
                if self.is_element_adr_regularized(p=p, k=k):
                    self.beta[p][k] = np.zeros(self.J[p][k])
                    self.beta_path[p][k] = np.zeros((self.lambda_n, self.J[p][k]))

                else:
                    # TODO: This should be optimized at some future point.
                    # Still need to think about a good setup for the link
                    # and the flat-cube problem
                    eta = self.distribution.link_function(theta[p], p)
                    dr = 1 / self.distribution.cube_to_flat(
                        self.distribution.link_inverse_derivative(eta, param=p), param=p
                    )
                    eta = self.distribution.cube_to_flat(eta, param=p)
                    dr = dr[:, k]

                    dl1dp1 = self.distribution.element_score(
                        y, theta=theta, param=p, k=k
                    )
                    dl2dp2 = self.distribution.element_hessian(
                        y, theta=theta, param=p, k=k
                    )

                    if self.distribution._scoring == "fisher":
                        wt = -(dl2dp2 / (dr * dr))
                        wv = eta[:, k] + dl1dp1 / (dr * wt)
                    elif self.distribution._scoring == "newton_rapson":
                        wt = -dl2dp2
                        wv = eta[:, k] + dl1dp1 / wt
                    else:
                        raise ValueError("Unknown scoring method.")
                    # Base Estimator abstracts the method away
                    # This will create inverted grams for OLS and
                    # non-inverted grams for LASSO
                    x = make_model_array(
                        X=X,
                        eq=self.equation[p][k],
                        fit_intercept=self.fit_intercept[p],
                    )
                    self.x_gram[p][k] = self.update_gram(
                        gram=self.x_gram_old[p][k],
                        x=x,
                        w=wt,
                        param=p,
                    )
                    self.y_gram[p][k] = self.update_y_gram(
                        gram=self.y_gram_old[p][k],
                        x=x,
                        y=wv,
                        w=wt,
                        param=p,
                    )
                    if self.method[p] == "lasso":
                        lambda_max = get_max_lambda(
                            self.x_gram[p][k],
                            self.y_gram[p][k],
                            self.is_regularized[p][k],
                        )
                        lambda_path = np.geomspace(
                            lambda_max, lambda_max * self.lambda_eps, self.lambda_n
                        )

                        self.beta_path[p][k], _ = online_coordinate_descent_path(
                            x_gram=self.x_gram[p][k],
                            y_gram=self.y_gram[p][k].squeeze(axis=-1),
                            beta_path=self.beta_path[p][k],
                            lambda_path=lambda_path,
                            is_regularized=self.is_regularized[p][k],
                            beta_lower_bound=np.repeat(-np.inf, self.J[p][k]),
                            beta_upper_bound=np.repeat(np.inf, self.J[p][k]),
                            which_start_value="previous_lambda",
                            selection="cyclic",
                            tolerance=1e-4,
                            max_iterations=1000,
                        )
                        eta_elem = x @ self.beta_path[p][k].T
                        theta_elem = self.distribution.element_link_inverse(
                            eta_elem, param=p, k=k, d=self.D
                        )

                        if self.ic == "max":
                            opt_ic = self.lambda_n - 1
                        else:
                            # Model selection
                            theta_ll = copy.deepcopy(theta)

                            theta_elem_delta = np.diff(theta_elem, axis=1)
                            theta_ms = self.distribution.set_theta_element(
                                theta_ll, theta_elem[:, 0], param=p, k=k
                            )
                            approx_ll = self.distribution.log_likelihood(
                                y, theta_ll
                            ).sum()
                            approx_ll = np.repeat(approx_ll, 100)
                            for l_idx in range(1, self.lambda_n):
                                theta_ms = self.distribution.set_theta_element(
                                    theta_ll, theta_elem[:, l_idx], param=p, k=k
                                )
                                approx_ll[l_idx] = (
                                    approx_ll[l_idx - 1]
                                    + (
                                        self.distribution.element_dl1_dp1(
                                            y, theta_ms, param=p, k=k
                                        )
                                        * theta_elem_delta[:, l_idx - 1]
                                    ).sum()
                                )
                            approx_ll = approx_ll + (
                                self.model_selection_old[p][k]["ll"]
                                * (1 - self.learning_rate)
                            )
                            # Count number of nonzero coefficients
                            # Subtract current beta if already fitted
                            # If in the first iteration, add intercept
                            # for all to-be-fitted parameters
                            # that are not AD-R regularized
                            nonzero = self.count_nonzero_coef(self.beta)
                            nonzero = -int(np.sum(self.beta[p][k] != 0))
                            if (outer_iteration == 0) & (inner_iteration == 0):
                                ## Add the "intercept" of the
                                ## next K - k parameters to the number of non-zero parameters
                                if self.adr_regularization[p] != 0:
                                    adr_zeroed = np.where(
                                        self.adr_distance[p]
                                        >= self.adr_regularization[p],
                                        0,
                                        1,
                                    )
                                else:
                                    adr_zeroed = np.ones(self.K[p])

                                nonzero = nonzero + np.sum(
                                    (np.ones(self.K[p]) * adr_zeroed)[k:]
                                )
                            ic = information_criteria_likelihood(
                                approx_ll, self.n_observations, nonzero, self.ic
                            )
                            opt_ic = np.argmin(ic)

                            self.model_selection[p][k] = {
                                "ll": approx_ll,
                                "non_zero": nonzero,
                                "ic": ic,
                            }
                        # Select the optimal beta
                        self.beta[p][k] = self.beta_path[p][k][opt_ic, :]
                        theta = self.distribution.set_theta_element(
                            theta, theta_elem[:, opt_ic], param=p, k=k
                        )
                    elif self.method[p] == "ols":
                        self.beta[p][k] = self.x_gram[p][k] @ self.y_gram[p][k]
                    else:
                        raise ValueError("Method not recognized")

                    eta[:, k] = np.squeeze(x @ self.beta[p][k])
                    theta[p] = self.distribution.link_inverse(
                        self.distribution.flat_to_cube(eta, param=p), param=p
                    )

                self.current_likelihood = (
                    self.distribution.log_likelihood(y, theta=theta).sum()
                    + self.old_likelihood_discounted
                )

            if inner_iteration == (self.max_iterations_inner - 1):
                warnings.warn(
                    "Reached max inner iterations. Algorithm may or may not be converged."
                )
            self.iteration_count[outer_iteration, p] = inner_iteration
        return theta

    # Different UV - MV
    def _outer_update(self, X, y, theta):

        for outer_iteration in range(self.max_iterations_outer):
            # Breaking criteria
            global_likelihood = self.current_likelihood
            if outer_iteration == 0:
                global_old_likelihood = 0
            if outer_iteration > 0:
                if (
                    np.abs(global_likelihood - global_old_likelihood)
                    / np.abs(global_likelihood)
                    < self.rel_tol_outer
                ):
                    break
                if (
                    np.abs(global_likelihood - global_old_likelihood)
                    < self.abs_tol_outer
                ):
                    break
            global_old_likelihood = global_likelihood

            for p in range(self.distribution.n_params):
                theta = self._inner_update(
                    X=X, y=y, theta=theta, outer_iteration=outer_iteration, p=p
                )
                if self.verbose >= 2:  #
                    print(
                        self._verbose_prefix,
                        f"Outer Iteration {outer_iteration}: Fitting Distribution parameter {p}",
                        end=self._verbose_end[2],
                    )

        self.theta = theta
