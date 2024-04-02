# Source: https://github.com/lunayht/DBALwithImgData 
import os
import numpy as np
from typing import Dict
from sklearn.base import BaseEstimator
from torch.utils.data import ConcatDataset, Dataset
from modAL.models.base import BaseLearner
from typing import Any, Callable, List, Optional, Tuple, Dict
from skorch.helper import SliceDataset
from UniversalFakeDetect.earlystop import EarlyStopping
from active_learning.acquisition_functions import uniform, max_entropy, bald, var_ratios, mean_std


class SliceActiveLearner(BaseLearner):
    """
    This class is an model of a general classic (machine learning) active learning algorithm.

    Args:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop,
            for instance, modAL.uncertainty.uncertainty_sampling.
        training: Initial training samples, if available.
        bootstrap_init: If initial training data is available, bootstrapping can be done during the first training.
            Useful when building Committee models with bagging.
        on_transformed: Whether to transform samples with the pipeline defined by the estimator
            when applying the query strategy.
        **fit_kwargs: keyword arguments.

    Attributes:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop.
        training: If the model hasn't been fitted yet it is None, otherwise it contains the samples
            which the model has been trained on. If provided, the method fit() of estimator is called during __init__()
    """

    def __init__(self,
                 estimator: BaseEstimator,
                 query_strategy,
                 training: Optional[Dataset] = None,
                 bootstrap_init: bool = False,
                 on_transformed: bool = False,
                 **fit_kwargs
                 ) -> None:
        super().__init__(estimator, query_strategy, on_transformed, **fit_kwargs)

        self.training = training

        if training is not None:
            self._fit_to_known(bootstrap=bootstrap_init, **fit_kwargs)


    def _add_training_data(self, dataset: Dataset) -> None:
        """
        Adds the new data and label to the known data, but does not retrain the model.

        Args:
            dataset: The new samples for which the labels are supplied by the expert.

        Note:
            If the classifier has been fitted, the features in X have to agree with the training samples which the
            classifier has seen.
        """
        if self.training is None:
            self.training = dataset
        else:
            try:
                self.training = ConcatDataset([self.training, dataset])
            except ValueError:
                raise ValueError('the dimensions of the new training data and label must'
                                 'agree with the training data and labels provided so far')


    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        """
        Fits self.estimator to the training data and labels provided to it so far.

        Args:
            bootstrap: If True, the method trains the model on a set bootstrapped from the known training instances.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Returns:
            self
        """
        X_training = SliceDataset(self.training, idx=0)
        y_training = np.asarray(SliceDataset(self.training, idx=1)).astype(np.float32)

        if not bootstrap:
            self.estimator.fit(X_training, y_training, **fit_kwargs)
        else:
            n_instances = len(X_training)
            bootstrap_idx = np.random.choice(
                range(n_instances), n_instances, replace=True)
            self.estimator.fit(
                X_training[bootstrap_idx], y_training[bootstrap_idx], **fit_kwargs)

        return self


    def fit(self, dataset: Dataset, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        """
        Interface for the fit method of the predictor. Fits the predictor to the supplied data, then stores it
        internally for the active learning loop.

        Args:
            dataset: The samples to be fitted.
            bootstrap: If true, trains the estimator on a set bootstrapped from X.
                Useful for building Committee models with bagging.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Note:
            When using scikit-learn estimators, calling this method will make the ActiveLearner forget all training data
            it has seen!

        Returns:
            self
        """
        self.training = dataset
        return self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)


    def teach(self, dataset:Dataset, bootstrap: bool = False, only_new: bool = False, **fit_kwargs) -> None:
        """
        Adds X and y to the known training data and retrains the predictor with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, training is done on a bootstrapped dataset. Useful for building Committee models
                with bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
                Useful when working with models where the .fit() method doesn't retrain the model from scratch (e. g. in
                tensorflow or keras).
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        if not only_new:
            self._add_training_data(dataset)
            self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)
        else:
            X = SliceDataset(dataset, idx=0)
            y = np.asarray(SliceDataset(dataset, idx=1)).astype(np.float32)
            self._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)


    def query(self, X_pool, *query_args, return_metrics: bool = False, **query_kwargs):
        """
        Finds the n_instances most informative point in the data provided by calling the query_strategy function.

        Args:
            X_pool: Pool of unlabeled instances to retrieve most informative instances from
            return_metrics: boolean to indicate, if the corresponding query metrics should be (not) returned
            *query_args: The arguments for the query strategy. For instance, in the case of
                :func:`~modAL.uncertainty.uncertainty_sampling`, it is the pool of samples from which the query strategy
                should choose instances to request labels.
            **query_kwargs: Keyword arguments for the query strategy function.

        Returns:
            Value of the query_strategy function. Should be the indices of the instances from the pool chosen to be
            labelled and the instances themselves. Can be different in other cases, for instance only the instance to be
            labelled upon query synthesis.
            query_metrics: returns also the corresponding metrics, if return_metrics == True
        """
        query_idx, query_instance = self.query_strategy(
            self, X_pool, *query_args, **query_kwargs)
        return query_idx, query_instance


    def score(self, X: SliceDataset, y: SliceDataset, **score_kwargs) -> Any:
        """
        Interface for the score method of the predictor.

        Args:
            X: The samples for which prediction accuracy is to be calculated.
            y: Ground truth labels for X.
            **score_kwargs: Keyword arguments to be passed to the .score() method of the predictor.

        Returns:
            The score of the predictor.
        """
        return self.estimator.score(X, np.asarray(y).astype(np.float32), **score_kwargs)


def adjust_learning_rate(estimator, min_lr=1e-6):
    for param_group in estimator.optimizer_.param_groups:
        param_group['lr'] /= 10.
    if param_group['lr'] < min_lr:
        return False
    return True


def active_learning_procedure(
    opt,
    query_strategy,
    init_dataset: Dataset,
    pool_dataset: Dataset,
    val_dataset: Dataset,
    test_datasets: Dict[str, Dataset],
    estimator,
    T: int = 100,
    n_query: int = 10,
    training: bool = True,
):
    """Active Learning Procedure

    Attributes:
        query_strategy: Choose between Uniform(baseline), max_entropy, bald, etc.
        val_dataset: Validation dataset
        pool_dataset: Query pool set
        init_dataset: Initial training set data points
        test_datsets: dict of Dataset objects, one for each test source
        estimator: Neural Network architecture, e.g. CNN
        T: Number of MC dropout iterations (repeat acqusition process T times)
        n_query: Number of points to query from X_pool
        training: If False, run test without MC Dropout (default: True)
    """
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    X_val, y_val = SliceDataset(val_dataset,idx=0), SliceDataset(val_dataset,idx=1)
    
    learner = SliceActiveLearner(
        estimator=estimator,
        training=init_dataset,
        query_strategy=query_strategy,
    )

    perf_hist = [learner.score(X_val, y_val)]

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    for index in range(T):
        query_idx, query_instance = learner.query(
            pool_dataset, n_query=n_query, T=T, training=training, opt=opt
        )
        learner.teach(query_instance)

        pool_dataset.remove_samples(query_idx)
        
        model_accuracy_val = learner.score(X_val, y_val)
        perf_hist.append(model_accuracy_val)

        # create model checkpoint
        if (index + 1) % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (index + 1))
            # save model weights
            learner.estimator.save_params(f_params=f"{save_dir}/dropout_iter_{(index + 1)}.pkl")
        
        # validate model
        if (index + 1) % opt.val_freq == 0:
            print(f"Val Accuracy after dropout iter {index+1}: {model_accuracy_val:0.4f}")
            
            early_stopping(model_accuracy_val, learner.estimator.module_)
            if early_stopping.early_stop:
                cont_train = adjust_learning_rate(learner.estimator)
                if cont_train:
                    print("Learning rate dropped by 10, continue training...")
                    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    print("Early stopping.")
                    break

    # save model weights
    learner.estimator.save_params(f_params=f"{save_dir}/dropout_iter_{(index + 1)}.pkl")

    model_accuracy_test = {}
    for k,dt in test_datasets.items():
        X_test, y_test = SliceDataset(dt,idx=0), SliceDataset(dt,idx=1)
        model_accuracy_test[k] = learner.score(X_test, y_test)
    print(f"********** Test Accuracy per experiment: {model_accuracy_test} **********")
    return perf_hist, model_accuracy_test


def select_acq_function(acq_func: int = 0) -> list:
    """Choose types of acqusition function

    Attributes:
        acq_func: 0-all(unif, max_entropy, bald), 1-unif, 2-maxentropy, 3-bald, \
                  4-var_ratios, 5-mean_std
    """
    acq_func_dict = {
        0: [uniform, max_entropy, bald, var_ratios, mean_std],
        1: uniform,
        2: max_entropy,
        3: bald,
        4: var_ratios,
        5: mean_std,
    }
    return acq_func_dict[acq_func]
