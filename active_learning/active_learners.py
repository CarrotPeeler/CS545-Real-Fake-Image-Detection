from typing import Any, Optional

import numpy as np
import torch
from modAL.models.base import BaseLearner
from torch.utils.data import ConcatDataset, Dataset, Subset

from UniversalFakeDetect.networks.trainer import Trainer
from active_learning.acquisition_functions import minmax_additive_norm 


class TorchActiveLearner(BaseLearner):
    """
    This class is an model of a general classic
    (machine learning) active learning algorithm.

    Args:
        model: The model to be used in the active learning loop.
        query_strategy: Function providing the
            query strategy for the active learning loop,
            for instance, modAL.uncertainty.uncertainty_sampling.
        train_dataset: Initial training samples, if available.
        bootstrap_init: If initial training data is available,
            bootstrapping can be done during the first training.
            Useful when building Committee models with bagging.
        on_transformed: Whether to transform samples
            with the pipeline defined by the model
            when applying the query strategy.
        **fit_kwargs: keyword arguments.

    Attributes:
        model: The model to be used in the active learning loop.
        query_strategy: Function providing the query
            strategy for the active learning loop.
        training: If the model hasn't been fitted yet
            it is None, otherwise it contains the samples
            which the model has been trained on. If provided,
            the method fit() of model is called during __init__()
    """

    def __init__(
        self,
        opt,
        model: Trainer,
        train_func,
        val_func,
        query_strategy,
        train_dataset: Optional[Dataset] = None,
        bootstrap_init: bool = False,
        on_transformed: bool = False,
        **fit_kwargs,
    ) -> None:
        super().__init__(model, query_strategy, on_transformed, **fit_kwargs)

        self.opt = opt
        self.model = model
        self.train_func = train_func
        self.val_func = val_func
        self.train_dataset = train_dataset

        if train_dataset is not None and opt.use_weighted_loss:
            self.acq_weights = torch.ones(len(train_dataset)).to(model.device)
            # map sample idx in dataset to corresponding weight idx in acq_weights
            self.acq_weights_idx_map = {k:k for k in range(len(train_dataset))} 
        else:
            self.acq_weights = None
            self.acq_weights_idx_map = None

        if train_dataset is not None:
            self._fit_to_known(epochs=opt.al_niter_warmup, bootstrap=bootstrap_init, **fit_kwargs)

    def train(self, epochs:int, dataset: Dataset):
        self.train_func(self.opt, epochs, self, dataset)

    def validate(self, dataset: Dataset):
        return self.val_func(self.opt, self, dataset)

    def get_query_weights(self, query_scores: np.ndarray):
        """
        Convert query scores to weight values for sample loss
        """
        query_weights = minmax_additive_norm(query_scores)
        query_weights = np.power(query_weights, 3)
        query_weights = torch.tensor(query_weights).to(self.model.device)
        return query_weights
    
    def regress_acq_weights(self, iter: int, decay_rate=0.99):
        """
        Perform exponential decay on weights for sample loss.
        Decay is based on how many iterations have passed. 
        """
        if iter > 0:
            self.acq_weights *= decay_rate
            self.acq_weights[self.acq_weights < 1] = 1
    
    def add_to_weights_idx_map(self, query_scores: np.ndarray, query_idxs: np.ndarray):
        """
        Maps a sample's dataset index to an index corresponding to where its weight value 
        is stored in the self.acq_weights array.
        """
        for i in range(len(query_idxs)):
            # shift idxs by init dataset size to avoid idx collision
            dataset_idx = query_idxs[i] + (self.opt.num_samples_per_class * 2) 
            # compute corresponding weight idx in self.acq_weights
            acq_weights_idx = len(self.acq_weights) - len(query_scores) + i
            # store mapping
            self.acq_weights_idx_map[dataset_idx] = acq_weights_idx        

    def _add_training_data(self, dataset: Dataset, query_scores=None, query_idxs=None) -> None:
        """
        Adds the new data and label to the known data,
        but does not retrain the model.

        Args:
            dataset: The new samples for which the labels
                are supplied by the expert.

        Note:
            If the classifier has been fitted, the features in X
            have to agree with the training samples which the
            classifier has seen.
        """
        if self.train_dataset is None:
            self.train_dataset = dataset

            if self.opt.use_weighted_loss:
                self.acq_weights = torch.ones(len(dataset)).to(self.model.device)
                self.acq_weights_idx_map = {k:k for k in range(len(dataset))}
        else:
            self.train_dataset = ConcatDataset([self.train_dataset, dataset])

            if self.opt.use_weighted_loss:
                assert len(query_scores) == len(query_idxs)
                # query here on self.train_dataset to reassign weight values to self.acq_weights
                query_weights = self.get_query_weights(query_scores)
                self.acq_weights = torch.cat([self.acq_weights, query_weights])
                self.add_to_weights_idx_map(query_scores, query_idxs)
        print(f"Updated Train Size: {len(self.train_dataset)}")

    def _fit_to_known(self, epochs:int, bootstrap: bool = False, **fit_kwargs) -> "BaseLearner":
        """
        Fits self.model to the training data and labels provided to it so far.

        Args:
            bootstrap: If True, the method trains the model on a set
            bootstrapped from the known training instances.
            **fit_kwargs: Keyword arguments to be passed to the
            fit method of the predictor.

        Returns:
            self
        """
        if not bootstrap:
            self.train(epochs, self.train_dataset)
        else:
            n_instances = len(self.train_dataset)

            bootstrap_idx = np.random.choice(range(n_instances), n_instances, replace=True)

            bootstrapped_dataset = Subset(self.train_dataset, bootstrap_idx)

            self.train(epochs, bootstrapped_dataset)

        return self

    def fit(self, epochs:int, dataset: Dataset, bootstrap: bool = False, **fit_kwargs) -> "BaseLearner":
        """
        Interface for the fit method of the predictor.
        Fits the predictor to the supplied data, then stores it
        internally for the active learning loop.

        Args:
            dataset: The samples to be fitted.
            bootstrap: If true, trains the model on a set bootstrapped from X.
                Useful for building Committee models with bagging.
            **fit_kwargs: Keyword arguments to be passed to
                the fit method of the predictor.

        Note:
            When using scikit-learn models, calling this method
            will make the ActiveLearner forget all training data
            it has seen!

        Returns:
            self
        """
        self.train_dataset = dataset
        return self._fit_to_known(epochs, bootstrap=bootstrap, **fit_kwargs)

    def _fit_on_new(
        self, epochs:int, dataset: Dataset, bootstrap: bool = False, **fit_kwargs
    ) -> "BaseLearner":
        """
        Fits self.estimator to the given data and labels.

        Args:
            dataset: The new samples for which the
            labels are supplied by the expert.
            bootstrap: If True, the method trains
            the model on a set bootstrapped from X.
            **fit_kwargs: Keyword arguments to be
            passed to the fit method of the predictor.

        Returns:
            self
        """
        if not bootstrap:
            self.train(epochs, dataset)
        else:
            n_instances = len(dataset)

            bootstrap_idx = np.random.choice(range(n_instances), n_instances, replace=True)

            bootstrapped_dataset = Subset(dataset, bootstrap_idx)

            self.train(epochs, bootstrapped_dataset)

        return self

    def teach(
        self,
        dataset: Dataset,
        query_scores=None,
        query_idxs=None,
        bootstrap: bool = False,
        only_new: bool = False,
        **fit_kwargs,
    ) -> None:
        """
        Adds X and y to the known training data and
        retrains the predictor with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, training is done on a bootstrapped dataset.
                Useful for building Committee models with bagging.
            only_new: If True, the model is retrained using only X and y,
                ignoring the previously provided examples.
                Useful when working with models where the .fit()
                method doesn't retrain the model from scratch (e. g. in
                tensorflow or keras).
            **fit_kwargs: Keyword arguments to be passed to
                the fit method of the predictor.
        """
        if not only_new:
            self._add_training_data(dataset, query_scores, query_idxs)
            self._fit_to_known(epochs=self.opt.niter, bootstrap=bootstrap, **fit_kwargs)
        else:
            self._fit_on_new(dataset, bootstrap=bootstrap, **fit_kwargs)

    def query(self, X_pool, *query_args, return_metrics: bool = False, **query_kwargs):
        """
        Finds the n_instances most informative point in
        the data provided by calling the query_strategy function.

        Args:
            X_pool: Pool of unlabeled instances to retrieve
                most informative instances from
            return_metrics: boolean to indicate, if
                the corresponding query metrics should be (not) returned
            *query_args: The arguments for the query strategy.
                For instance, in the case of
                :func:`~modAL.uncertainty.uncertainty_sampling`, it is
                the pool of samples from which the query strategy
                should choose instances to request labels.
            **query_kwargs: Keyword arguments for the query strategy function.

        Returns:
            Value of the query_strategy function. Should be the indices of
            the instances from the pool chosen to be
            labelled and the instances themselves. Can be different in
            other cases, for instance only the instance to be
            labelled upon query synthesis.
            query_metrics: returns also the corresponding metrics,
            if return_metrics == True
        """
        query_idx, query_instance, query_score = self.query_strategy(
            self, X_pool, *query_args, **query_kwargs
        )
        return query_idx, query_instance, query_score

    def score(self, dataset: Dataset, **score_kwargs) -> Any:
        """
        Interface for the score method of the predictor.

        Args:
            dataset: The samples for which prediction accuracy is to be
                calculated.
            **score_kwargs: Keyword arguments to be passed to the .score()
                method of the predictor.

        Returns:
            The score of the predictor.
        """
        return self.validate(dataset)

    def forward(self, dataset: Dataset, training: bool):
        """
        Performs forward pass of the model on a dataset
        Only use for Active Learning to perform inference with
        dropout layers enabled for Monte Carlo Dropout

        args:
            dataset: Torch Dataset
            training: whether to enable dropout layers for MC Dropout
        """
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_threads,
        )
        with torch.set_grad_enabled(training):
            self.model.train(training)

            logits = []
            targets = []
            # perform mini-batch inference
            for i, data in enumerate(loader):
                self.model.set_input(data)
                out = self.model.forward_raw()
                logits.append(out)
                targets.append(data[1])

            logits = torch.cat(logits)
            targets = torch.cat(targets)
        return logits, targets
