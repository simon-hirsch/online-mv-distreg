import numpy as np
import scipy.stats as st
from ondil.information_criteria import InformationCriterion
from sklearn.base import BaseEstimator
from sklearn.covariance import graphical_lasso


def get_v_indices(d, rank):
    idx1 = np.concatenate([np.arange(d)] * rank)
    idx2 = np.concatenate([np.full(d, r) for r in range(rank)])
    return (idx1, idx2)


class OnlineGaussianCopula(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, uniform):

        self.n_observations = uniform.shape[0]
        self.D = uniform.shape[1]

        gauss = st.norm().ppf(uniform)
        self.cov = np.cov(gauss, rowvar=False)
        self.loc = np.mean(gauss, axis=0)

    def _step_update(self, uniform):
        if not uniform.shape[0] == 1:
            raise ValueError("shape should 1 x D")

        self.n_observations += 1
        gauss = st.norm().ppf(uniform).squeeze()
        self.loc = (1 / self.n_observations) * (
            (self.n_observations - 1) * self.loc + gauss
        )
        self.cov = (1 / self.n_observations) * (
            (self.n_observations - 1) * self.cov + np.outer(gauss, gauss)
        )

    def update(self, uniform: np.ndarray):
        if uniform.shape[0] == 1:
            self._step_update(uniform)
        else:
            for i in range(uniform.shape[0]):
                self._step_update(uniform[[i], :])

    def sample(self, shape):
        samples = st.norm().cdf(st.multivariate_normal(self.loc, self.cov).rvs(shape))
        return samples


class OnlineSparseGaussianCopula(BaseEstimator):

    def __init__(self, lambda_eps=1e-3, lambda_n=50, ic="bic"):
        self.lambda_eps = lambda_eps
        self.lambda_n = lambda_n
        self.ic = ic
        self.cd_params = {"max_iter": 500, "tol": 1e-3}

    def fit(self, uniform):

        self.n_observations = uniform.shape[0]
        self.D = uniform.shape[1]

        gauss = st.norm().ppf(uniform)
        self.cov = np.cov(gauss, rowvar=False)
        self.loc = np.mean(gauss, axis=0)

        # Run regularization
        self.lambda_max = self.cov[np.triu_indices(self.D, k=1)].max()
        self.lambda_grid = np.linspace(
            self.lambda_max, self.lambda_max * self.lambda_eps, self.lambda_n
        )
        self.reg_cov = np.zeros((self.lambda_n, *self.cov.shape))
        self.reg_cov[0] = self.cov
        self.reg_ll = np.zeros(self.lambda_n)

        for a, regularization in enumerate(self.lambda_grid):
            self.reg_cov[a] = graphical_lasso(
                self.cov, alpha=regularization, **self.cd_params
            )[0]
            self.reg_ll[a] = (
                st.multivariate_normal(self.loc, self.reg_cov[a]).logpdf(gauss).sum()
            )

        self.n_parameters = np.sum(self.reg_cov != 0, axis=(-2, -1))

        self.information_criteria = InformationCriterion(
            n_observations=self.n_observations,
            n_parameters=self.n_parameters,
            criterion=self.ic,
        ).from_ll(self.reg_ll)
        self.opt_ic = np.argmin(self.information_criteria)
        self.opt_cov = self.reg_cov[self.opt_ic]

    def _step_update(self, uniform):
        if not uniform.shape[0] == 1:
            raise ValueError("shape should 1 x D")

        self.n_observations += 1
        gauss = st.norm().ppf(uniform).squeeze()
        self.loc = (1 / self.n_observations) * (
            (self.n_observations - 1) * self.loc + gauss
        )
        self.cov = (1 / self.n_observations) * (
            (self.n_observations - 1) * self.cov + np.outer(gauss, gauss)
        )

        # Run regularization
        self.lambda_max = self.cov[np.triu_indices(self.D, k=1)].max()
        self.lambda_grid = np.linspace(
            self.lambda_max, self.lambda_max * self.lambda_eps, self.lambda_n
        )
        self.reg_cov = np.zeros((self.lambda_n, *self.cov.shape))
        self.reg_cov[0] = self.cov

        for a, regularization in enumerate(self.lambda_grid):
            self.reg_cov[a] = graphical_lasso(
                self.cov, alpha=regularization, **self.cd_params
            )[0]
            self.reg_ll[a] += (
                st.multivariate_normal(self.loc, self.reg_cov[a]).logpdf(gauss).sum()
            )

        self.n_parameters = np.sum(self.reg_cov != 0, axis=(-2, -1))
        self.information_criteria = InformationCriterion(
            n_observations=self.n_observations,
            n_parameters=self.n_parameters,
            criterion=self.ic,
        ).from_ll(self.reg_ll)
        self.opt_ic = np.argmin(self.information_criteria)
        self.opt_cov = self.reg_cov[self.opt_ic]

    def update(self, uniform: np.ndarray):
        if uniform.shape[0] == 1:
            self._step_update(uniform)
        else:
            for i in range(uniform.shape[0]):
                self._step_update(uniform[[i], :])

    def sample(self, shape):
        samples = st.norm().cdf(
            st.multivariate_normal(self.loc, self.opt_cov).rvs(shape)
        )
        return samples


# Copula likelihood
def gaussian_copula_log_likelihood(
    uniform: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    # Naive estimation as in the paper by, see pages 106/107
    # @article{arbenz2013bayesian,
    #     title={Bayesian copulae distributions, with application to operational risk management—some comments},
    #     author={Arbenz, Philipp},
    #     journal={Methodology and computing in applied probability},
    #     volume={15},
    #     pages={105--108},
    #     year={2013},
    #     publisher={Springer}
    # }

    H = uniform.shape[1]
    std = cov[:, range(H), range(H)] ** 0.5
    corr = cov / (std[..., None] @ std[:, None, :])
    prec = np.linalg.inv(corr) - np.diag(np.ones(H))[None, ...]
    transf_unif = st.norm().ppf(np.clip(uniform, 1e-10, 1 - 1e-10))
    term_1 = (
        -1 / 2 * (transf_unif[:, None, :] @ prec @ transf_unif[:, :, None]).squeeze()
    )
    term_2 = 1 / np.sqrt(np.linalg.det(corr))
    return np.log(term_2) + term_1
