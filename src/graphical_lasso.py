from typing import Tuple

import numba as nb
import numpy as np
from rolch.coordinate_descent import soft_threshold


@nb.jit()
def coordinate_descent_glasso(
    beta: np.ndarray,
    V: np.ndarray,
    s: np.ndarray,
    regularization: float,
    max_iter: int = 1000,
    tolerance: float = 1e-4,
) -> np.ndarray:
    J = beta.shape[0]
    index = np.arange(J)
    beta_star = np.copy(beta)
    for lasso_iter in range(max_iter):
        beta_old = np.copy(beta_star)
        for j in range(J):
            k = index != j
            beta_star[j] = soft_threshold(
                s[j] - 2 * (V[k, j] @ beta_star[k]), regularization
            ) / (2 * V[j, j])
        if np.linalg.norm(np.abs(beta_star - beta_old)) < tolerance:
            break
    return beta_star


def graphical_lasso(
    cov: np.ndarray,
    regularization: float = 0,
    cov_tolerance: float = 1e-3,
    cov_iterations: int = 100,
    lasso_tolerance: float = 1e-3,
    lasso_iterations: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """The graphical lasso as in Friedmann et al (2008) for estimating sparse covariance matrices.

    Args:
        cov (np.ndarray): Empiricial covariance matrix
        regularization (float, optional): Regularization Parameter. Defaults to 0.
        cov_tolerance (float, optional): Tolerance for the covariance. Defaults to 1e-3.
        cov_iterations (int, optional): Max iterations. Defaults to 100.
        lasso_tolerance (float, optional): Tolerance for the lasso. Defaults to 1e-3.
        lasso_iterations (int, optional): Max iterations for the lasso. Defaults to 100.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns the sparse covariance matrix W and the precision matrix $\Theta$.
    """

    D = cov.shape[0]
    W = np.copy(cov)
    indices = np.arange(D)

    B = np.zeros_like(cov)
    P = np.zeros_like(cov)

    for cov_iter in range(cov_iterations):
        cov_old = np.copy(W)
        for p in range(D):
            S12 = cov[p, indices != p]
            W11 = np.delete(np.delete(W, p, 0), p, 1)
            V = np.copy(W11)
            beta = np.copy(S12)
            beta = np.zeros_like(S12)
            beta = coordinate_descent_glasso(
                beta,
                V,
                s=S12,
                regularization=regularization,
                tolerance=lasso_tolerance,
                max_iter=lasso_iterations,
            )
            W[p, indices != p] = 2 * W11 @ beta
            W[indices != p, p] = 2 * W11 @ beta

            B[p, indices != p] = beta
            B[indices != p, p] = beta

            # Update the precision matrix!
            P[p, p] = 1 / (W[p, p] - 2 * W[indices != p, p] @ beta)
            P[p, indices != p] = -2 * P[p, p] * beta
            P[indices != p, p] = -2 * P[p, p] * beta

            gap = np.sum(cov * P)
            gap -= P.shape[0]
            gap += regularization * (np.abs(P).sum() - np.abs(np.diag(P)).sum())

        if np.abs(gap) < cov_tolerance:  # & (cov_iter > 5):
            print(f"breaking after {cov_iter} iterations, dual_gap {gap}")
            break

    return W, P
