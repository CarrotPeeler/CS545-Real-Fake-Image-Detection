# Modified from: https://github.com/lunayht/DBALwithImgData
# Changed implementation to handle torch.utils.data.DataLoader class instead of Numpy array and binary classification

import torch
import numpy as np
from scipy import stats
from tqdm.auto import tqdm
from torch.utils.data import Dataset, Subset


def predictions_from_pool(
    learner, X_pool: Dataset, opt, pool_idxs:np.ndarray, T: int = 100, training: bool = True
):
    """Run random_subset prediction on model and return the output

    Attributes:
        X_pool: Pool set to select uncertainty,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    random_subset_idxs = np.random.choice(pool_idxs, size=10000, replace=False)
    subset = Subset(X_pool, random_subset_idxs)
    with torch.no_grad():
        outputs = np.stack(
            [
                torch.softmax(
                    learner.forward(subset, training=training),
                    dim=-1,
                )
                .cpu()
                .numpy()
                for _ in range(T)
            ]
        )
    outputs = np.squeeze(outputs, axis=-1)
    return outputs, random_subset_idxs


def uniform(
    learner, X_pool: Dataset, opt, pool_idxs:np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Baseline acquisition a(x) = unif() with unif() a function
    returning a draw from a uniform distribution over the interval [0,1].
    Using this acquisition function is equivalent to choosing a point
    uniformly at random from the pool.

    Attributes:
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that randomly select from pool set,
        training: If False, run test without MC dropout. (default=True)
    """
    query_idx = np.random.choice(pool_idxs, size=n_query, replace=False)
    subset = Subset(X_pool, query_idx)
    return query_idx, subset


def shannon_entropy_function(
    learner, X_pool: Dataset, opt, pool_idxs:np.ndarray, T: int = 100, E_H: bool = False, training: bool = True
):
    """H[y|x,D_train] := - sum_{c} p(y=c|x,D_train)log p(y=c|x,D_train)

    Attributes:
        learner: learner that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        T: Number of MC dropout iterations aka training iterations,
        E_H: If True, compute H and EH for BALD (default: False),
        training: If False, run test without MC dropout. (default=True)
    """
    outputs, random_subset = predictions_from_pool(learner, X_pool, opt, pool_idxs, T, training=training)
    pc = outputs.mean(axis=0)
    # Binary Shannon Entropy
    H = (-pc * np.log(pc + 1e-10) - (1 - pc) * np.log(1 - pc + 1e-10)) # To avoid division with zero, add 1e-10
    if E_H:
        # Binary Model Entropy 
        E = -np.mean(outputs * np.log(outputs + 1e-10) + 
                       (1 - outputs) * np.log(1 - outputs + 1e-10), axis=0)
        return H, E, random_subset
    return H, random_subset


def max_entropy(
    learner, X_pool: Dataset, opt, pool_idxs:np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Choose pool points that maximise the predictive entropy.
    Using Shannon entropy function.

    Attributes:
        learner: learner that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise max_entropy a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    acquisition, random_subset = shannon_entropy_function(
        learner, X_pool, opt, pool_idxs, T, training=training
    )
    idx = (-acquisition).argsort()[:n_query] # retrieve n highest entropy sample idxs
    query_idx = random_subset[idx] # fetch pool idxs that correspond to the n sample idxs from the random subset
    subset = Subset(X_pool, query_idx)
    return query_idx, subset


def bald(
    learner, X_pool: Dataset, opt, pool_idxs:np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Choose pool points that are expected to maximise the information
    gained about the model parameters, i.e. maximise the mutal information
    between predictions and model posterior. Given
    I[y,w|x,D_train] = H[y|x,D_train] - E_{p(w|D_train)}[H[y|x,w]]
    with w the model parameters (H[y|x,w] is the entropy of y given w).
    Points that maximise this acquisition function are points on which the
    model is uncertain on average but there exist model parameters that produce
    disagreeing predictions with high certainty. This is equivalent to points
    with high variance in th einput to the softmax layer

    Attributes:
        learner: learner that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise bald a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    H, E_H, random_subset = shannon_entropy_function(
        learner, X_pool, opt, pool_idxs, T, E_H=True, training=training
    )
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset


def var_ratios(
    learner, X_pool: Dataset, opt, pool_idxs:np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Like Max Entropy but Variational Ratios measures lack of confidence.
    Given: variational_ratio[x] := 1 - max_{y} p(y|x,D_{train})

    Attributes:
        learner: learner that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise var_ratios a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    outputs, random_subset = predictions_from_pool(learner, X_pool, opt, pool_idxs, T, training)
    # get binary preds
    preds = (outputs[:, :] > 0.5).astype('uint8')
    _, count = stats.mode(preds, axis=0, keepdims=False)
    acquisition = (1 - count / T).reshape((-1,))
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset


def mean_std(
    learner, X_pool: Dataset, opt, pool_idxs:np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Maximise mean standard deviation
    Given: sigma_c = sqrt(E_{q(w)}[p(y=c|x,w)^2]-E_{q(w)}[p(y=c|x,w)]^2)

    Attributes:
        learner: learner that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise mean std a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    outputs, random_subset = predictions_from_pool(learner, X_pool, opt, pool_idxs, T, training)
    outputs_pos = outputs
    outputs_neg = 1 - outputs_pos

    # create separate pos and neg class probs in last dim
    probs = np.stack([outputs_neg, outputs_pos], axis=-1)

    sigma_c = np.std(probs, axis=0)
    acquisition = np.mean(sigma_c, axis=-1)
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset
