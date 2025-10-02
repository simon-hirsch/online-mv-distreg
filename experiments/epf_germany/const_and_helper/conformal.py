# See https://github.com/aangelopoulos/conformal-time-series
# See https://github.com/bruale/PefCodeBench
# Only removed the tqdm progress bar for the sake of simplicity

import numpy as np


# %%
#
def aci_clipped(scores, alpha, lr, window_length, T_burnin, ahead, *args, **kwargs):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in range(T_test):
        t_pred = t - ahead + 1
        clip_value = (
            scores[max(t_pred - window_length, 0) : t_pred].max()
            if t_pred > 0
            else np.inf
        )
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1 / (t_pred + 1):
                qs[t] = np.inf
            else:
                qs[t] = np.quantile(
                    scores[max(t_pred - window_length, 0) : t_pred],
                    1 - np.clip(alphat, 0, 1),
                    method="higher",
                )
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1 - alpha
            alphat = alphat - lr * grad

            if t < T_test - 1:
                alphas[t + 1] = alphat
        else:
            if t_pred > np.ceil(1 / alpha):
                qs[t] = np.quantile(scores[:t_pred], 1 - alpha)
            else:
                qs[t] = np.inf
        if qs[t] == np.inf:
            qs[t] = clip_value
    results = {"method": "ACI (clipped)", "q": qs, "alpha": alphas}
    return results
