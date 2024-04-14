# Modified from: https://github.com/lunayht/DBALwithImgData
# Changed implementation to handle torch.utils.data.DataLoader
# class instead of Numpy array and binary classification

import numpy as np
import torch
from torch.nn.modules.loss import BCELoss
from scipy import stats
from torch.utils.data import Dataset, Subset


def predictions_from_pool(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
):
    """Run random_subset prediction on model and return the output

    Attributes:
        X_pool: Pool set to select uncertainty,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    if subsample_size is None:
        random_subset_idxs = np.array(list(range(len(X_pool))))
        subset = X_pool
    else:
        random_subset_idxs = np.random.choice(pool_idxs, size=subsample_size, replace=False)
        subset = Subset(X_pool, random_subset_idxs)
        
    with torch.no_grad():
        probs_per_dropout_iter = []
        for _ in range(T):
            logits, targets = learner.forward(subset, training=training)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_per_dropout_iter.append(probs)
        outputs = np.stack(probs_per_dropout_iter)
    outputs = np.squeeze(outputs, axis=-1)
    return outputs, random_subset_idxs, targets


def uniform(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
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
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    T: int = 100,
    E_H: bool = False,
    subsample_size=10000,
    training: bool = True,
):
    """H[y|x,D_train] := - sum_{c} p(y=c|x,D_train)log p(y=c|x,D_train)

    Attributes:
        learner: learner that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        T: Number of MC dropout iterations aka training iterations,
        E_H: If True, compute H and EH for BALD (default: False),
        training: If False, run test without MC dropout. (default=True)
    """
    outputs, random_subset, targets = predictions_from_pool(
        learner, X_pool, opt, pool_idxs, T, subsample_size, training=training
    )
    pc = outputs.mean(axis=0)
    # Binary Shannon Entropy
    H = -pc * np.log(pc + 1e-10) - (1 - pc) * np.log(1 - pc + 1e-10)
    # To avoid division with zero, add 1e-10
    if E_H:
        # Binary Model Entropy
        E = -np.mean(
            outputs * np.log(outputs + 1e-10) + (1 - outputs) * np.log(1 - outputs + 1e-10), axis=0
        )
        return H, E, random_subset, targets
    return H, random_subset, targets


def max_entropy(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
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
    acquisition, random_subset, targets = shannon_entropy_function(
        learner, X_pool, opt, pool_idxs, T, subsample_size=subsample_size, training=training
    )
    if opt.balance_acquisition:
        idx = balance_acquisition(acquisition, targets, n_query)
    else:
        idx = (-acquisition).argsort()[:n_query]
    # retrieve n highest entropy sample idxs
    query_scores = acquisition[idx]
    # fetch pool idxs that correspond to the
    # n sample idxs from the random subset
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset, query_scores


def loss_weighted_max_entropy(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
):
    """
    Modified version of Max Entropy.
    Weighs each sample's uncertainty score by their individual loss.
    Performs Min-Max Additive 1 normalization on the Shannon Entropy scores
    to prevent uncertainty scores of 0 from not being influenced by high loss. 
    Then, multiplies normalized uncertainty scores by individual loss directly.
    """
    outputs, random_subset, targets = predictions_from_pool(
        learner, X_pool, opt, pool_idxs, T, subsample_size, training=training
    )
    pc = outputs.mean(axis=0)
    # Binary Shannon Entropy
    acquisition = -pc * np.log(pc + 1e-10) - (1 - pc) * np.log(1 - pc + 1e-10)
    # compute loss
    loss_fn = BCELoss(reduction="none")
    pc = torch.from_numpy(pc)
    loss = loss_fn(pc, targets.float()).detach().cpu().numpy()
    # compute weighted acquisition (uncertainty) using loss
    acquisition = loss * minmax_additive_norm(acquisition)
    # compute max entropy
    if opt.balance_acquisition:
        idx = balance_acquisition(acquisition, targets, n_query)
    else:
        idx = (-acquisition).argsort()[:n_query]
    # retrieve n highest entropy sample idxs
    query_scores = acquisition[idx]
    # fetch pool idxs that correspond to the
    # n sample idxs from the random subset
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset, query_scores


def bald(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
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
    H, E_H, random_subset, targets = shannon_entropy_function(
        learner,
        X_pool,
        opt,
        pool_idxs,
        T,
        E_H=True,
        subsample_size=subsample_size,
        training=training,
    )
    acquisition = H - E_H
    if opt.balance_acquisition:
        idx = balance_acquisition(acquisition, targets, n_query)
    else:
        idx = (-acquisition).argsort()[:n_query]
    query_scores = acquisition[idx]
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset, query_scores


def loss_weighted_bald(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
):
    """
    Modified version of BALD that weighs uncertainty scores via sample loss
    """
    outputs, random_subset, targets = predictions_from_pool(
        learner, X_pool, opt, pool_idxs, T, subsample_size, training=training
    )
    pc = outputs.mean(axis=0)
    # Binary Shannon and BALD Entropy
    H = -pc * np.log(pc + 1e-10) - (1 - pc) * np.log(1 - pc + 1e-10)
    E_H = -np.mean(
        outputs * np.log(outputs + 1e-10) + (1 - outputs) * np.log(1 - outputs + 1e-10), axis=0
    )
    acquisition = H - E_H
    # compute loss
    loss_fn = BCELoss(reduction="none")
    pc = torch.from_numpy(pc)
    loss = loss_fn(pc, targets.float()).detach().cpu().numpy()
    # compute weighted acquisition (uncertainty) using loss
    acquisition = loss * minmax_additive_norm(acquisition)
    if opt.balance_acquisition:
        idx = balance_acquisition(acquisition, targets, n_query)
    else:
        idx = (-acquisition).argsort()[:n_query]
    query_scores = acquisition[idx]
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset, query_scores


def var_ratios(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
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
    outputs, random_subset, targets = predictions_from_pool(
        learner, X_pool, opt, pool_idxs, T, subsample_size, training
    )
    # get binary preds
    preds = (outputs[:, :] > 0.5).astype("uint8")
    _, count = stats.mode(preds, axis=0, keepdims=False)
    acquisition = (1 - count / T).reshape((-1,))
    if opt.balance_acquisition:
        idx = balance_acquisition(acquisition, targets, n_query)
    else:
        idx = (-acquisition).argsort()[:n_query]
    query_scores = acquisition[idx]
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset, query_scores


def loss_weighted_var_ratios(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
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
    outputs, random_subset, targets = predictions_from_pool(
        learner, X_pool, opt, pool_idxs, T, subsample_size, training
    )
    # get binary preds
    preds = (outputs[:, :] > 0.5).astype("uint8")
    _, count = stats.mode(preds, axis=0, keepdims=False)
    acquisition = (1 - count / T).reshape((-1,))
    # compute loss
    loss_fn = BCELoss(reduction="none")
    pc = torch.from_numpy(pc)
    loss = loss_fn(pc, targets.float()).detach().cpu().numpy()
    # compute weighted acquisition (uncertainty) using loss
    acquisition = loss * minmax_additive_norm(acquisition)
    if opt.balance_acquisition:
        idx = balance_acquisition(acquisition, targets, n_query)
    else:
        idx = (-acquisition).argsort()[:n_query]
    query_scores = acquisition[idx]
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset, query_scores


def mean_std(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
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
    outputs, random_subset, targets = predictions_from_pool(
        learner, X_pool, opt, pool_idxs, T, subsample_size, training
    )
    outputs_pos = outputs
    outputs_neg = 1 - outputs_pos

    # create separate pos and neg class probs in last dim
    probs = np.stack([outputs_neg, outputs_pos], axis=-1)

    sigma_c = np.std(probs, axis=0)
    acquisition = np.mean(sigma_c, axis=-1)
    if opt.balance_acquisition:
        idx = balance_acquisition(acquisition, targets, n_query)
    else:
        idx = (-acquisition).argsort()[:n_query]
    query_scores = acquisition[idx]
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset, query_scores


def loss_weighted_mean_std(
    learner,
    X_pool: Dataset,
    opt,
    pool_idxs: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    subsample_size=10000,
    training: bool = True,
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
    outputs, random_subset, targets = predictions_from_pool(
        learner, X_pool, opt, pool_idxs, T, subsample_size, training
    )
    outputs_pos = outputs
    outputs_neg = 1 - outputs_pos

    # create separate pos and neg class probs in last dim
    probs = np.stack([outputs_neg, outputs_pos], axis=-1)

    sigma_c = np.std(probs, axis=0)
    acquisition = np.mean(sigma_c, axis=-1)
    
    # compute loss
    loss_fn = BCELoss(reduction="none")
    pc = torch.from_numpy(pc)
    loss = loss_fn(pc, targets.float()).detach().cpu().numpy()
    # compute weighted acquisition (uncertainty) using loss
    acquisition = loss * minmax_additive_norm(acquisition)

    if opt.balance_acquisition:
        idx = balance_acquisition(acquisition, targets, n_query)
    else:
        idx = (-acquisition).argsort()[:n_query]
    query_scores = acquisition[idx]
    query_idx = random_subset[idx]
    subset = Subset(X_pool, query_idx)
    return query_idx, subset, query_scores


def balance_acquisition(acquisition: np.ndarray, targets: torch.Tensor, n_query):
    """
    Ensures the query results have an even number of positive and negative class samples.
    Returns query indices for samples with high uncertainty scores. 
    """
    # dedicate half of the query size to each class
    n_query = int(n_query/2) 
    targets = targets.detach().cpu().numpy()
    # parse positive and negative class samples
    pos_cls_idxs = np.where(targets == 1)[0]
    neg_cls_idxs = np.where(targets == 0)[0]
    # parse acquisition based on class
    pos_acq = acquisition[pos_cls_idxs]
    neg_acq = acquisition[neg_cls_idxs]
    # get indices for samples w/ highest uncertainty 
    pos_query_idxs = (-pos_acq).argsort()[:n_query]
    neg_query_idxs = (-neg_acq).argsort()[:n_query]
    # concat both idxs
    query_idxs = np.concatenate([pos_query_idxs, neg_query_idxs])
    return query_idxs


def minmax_additive_norm(query_scores):
    """Perform Min-Max Normalization w/ additive 1 constant"""
    wl = query_scores
    # 1e-10 prevents division by zero
    norm = ((wl - min(wl.min(), 0.1)) / (wl.max() - min(wl.min(), 0.1) + 1e-10)) + 1 
    return norm
