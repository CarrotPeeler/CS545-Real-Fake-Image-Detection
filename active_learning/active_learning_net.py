# Source: https://github.com/lunayht/DBALwithImgData

import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import UniversalFakeDetect.distributed as du
from active_learning.acquisition_functions import *
from active_learning.active_learners import TorchActiveLearner
from UniversalFakeDetect.data import create_dataloader, shuffle_dataset
from UniversalFakeDetect.earlystop import EarlyStopping
from UniversalFakeDetect.networks.trainer import Trainer
from UniversalFakeDetect.validate import validate as val


def train(opt, learner: TorchActiveLearner, train_dataset: Dataset):
    """Generic PyTorch train function"""
    train_loader = create_dataloader(opt, premade_dataset=train_dataset)

    learner.model.train()

    if len(opt.gpu_ids) > 1:
        du.init_distributed_training(len(opt.gpu_ids), opt.shard_id)

    # run training epochs
    for epoch in tqdm(range(opt.niter)):
        ep_s_t = time.time()

        # set current epoch for the data loader
        if len(opt.gpu_ids) > 1:
            shuffle_dataset(train_loader, epoch)
            if hasattr(train_loader.dataset, "_set_epoch_num"):
                train_loader.dataset._set_epoch_num(epoch)

        if opt.use_weighted_loss:
            # perform decay on sample weights for loss
            learner.regress_acq_weights(epoch)

        # perform mini-batch training
        for i, data in enumerate(train_loader):
            learner.model.total_steps += 1

            learner.model.set_input(data[:2]) # [imgs, labels]
            # print(f"INDICES: {data[2]}")

            if opt.use_weighted_loss:
                assert learner.acq_weights is not None, "Error: weights for loss are of type None"
                assert learner.acq_weights_idx_map is not None, "Error: weights_idx_map is of type None"
                weights_idxs = list(map(lambda x:learner.acq_weights_idx_map[x.item()], data[2]))
                # s_idx, e_idx = opt.batch_size * i, opt.batch_size * (i + 1)
                # print(f"{s_idx} to {e_idx}\nWeights: {weights[weights_idxs]}")
                learner.model.optimize_parameters(learner.acq_weights[weights_idxs])
            else:
                learner.model.optimize_parameters()

            # sync GPUs
            torch.cuda.synchronize()

        # in case of fragmented memory
        torch.cuda.empty_cache()

        print(
            f"Epoch {epoch + 1} | Train loss: {learner.model.loss} | Dur: {(time.time() - ep_s_t):0.4f}s"
        )


def validate(opt, learner: TorchActiveLearner, val_dataset: Dataset):
    learner.model.eval()
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads
    )
    ap, r_acc, f_acc, acc = val(learner.model.model, val_loader, gpu_id=learner.model.device)
    if len(opt.gpu_ids) > 1:
        ap, r_acc, f_acc, acc = du.all_reduce([ap, r_acc, f_acc, acc])
    return acc, r_acc, f_acc, ap


def active_learning_procedure(
    opt,
    query_strategy,
    init_dataset: Dataset,
    pool_dataset: Dataset,
    val_dataset: Dataset,
    test_datasets: Dict[str, Dataset],
    model: Trainer,
    T: int = 100,
    n_iters: int = 10,
    n_query: int = 10,
    subsample_size=10000,
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
        n_iters: Number of active learning iterations
        n_query: Number of points to query from X_pool
        training: If False, run test without MC Dropout (default: True)
    """
    print("Performing Training on Initial Dataset...")
    learner = TorchActiveLearner(
        opt=opt,
        model=model,
        train_dataset=init_dataset,
        query_strategy=query_strategy,
        train_func=train,
        val_func=validate,
    )
    
    # track already queried idxs
    pool_idxs = np.array(range(len(pool_dataset)))

    v_s_t = time.time()
    perf_hist = [learner.score(val_dataset)]
    v_e_t = time.time()
    print(
        f"Validation Time: {(v_e_t - v_s_t):0.4f}s\
          | Acc: {perf_hist[0][0]:0.4f}\
          | R_Acc: {perf_hist[0][1]:0.4f}\
          | F_Acc: {perf_hist[0][2]:0.4f}\
          | AP: {perf_hist[0][3]:0.4f}"
    )

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    for index in tqdm(range(n_iters)):
        print(f"\nDropout Iter: {index}")

        q_s_t = time.time()
        query_idxs, query_instance, query_scores = learner.query(
            pool_dataset,
            opt=opt,
            pool_idxs=pool_idxs,
            n_query=n_query,
            T=T,
            subsample_size=subsample_size,
            training=training,
        )
        q_e_t = time.time()
        print(f"QUERY TIME: {(q_e_t - q_s_t):0.4f}s")

        # train model over new queried data
        learner.teach(query_instance, query_scores, query_idxs)

        # remove queried data from pool
        remove_idxs = np.concatenate([np.where(pool_idxs == x)[0] for x in query_idxs])
        pool_idxs = np.delete(pool_idxs, remove_idxs)
        print(f"Updated Pool Size: {len(pool_idxs)}")

        # create model checkpoint
        if index % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" % index)
            # save model weights
            learner.model.save_networks(f"dropout_iter_{index}.pth")

        # validate model
        if index % opt.val_freq == 0:
            acc, r_acc, f_acc, ap = learner.score(val_dataset)
            perf_hist.append((acc, ap))

            print(
                f"Val Acc: {acc:0.4f}\
                  | R_Acc: {r_acc:0.4f}\
                  | F_Acc: {f_acc:0.4f}\
                  | AP: {ap:0.4f}"
            )

            if opt.al_earlystop_metric == 'acc':
                earlystop_metric = acc
            elif opt.al_earlystop_metric == 'ap':
                earlystop_metric = ap
                
            early_stopping(earlystop_metric, learner.model)

            if early_stopping.early_stop:
                cont_train = learner.model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training...")
                    early_stopping = EarlyStopping(
                        patience=opt.earlystop_epoch, delta=-0.002, verbose=True
                    )
                else:
                    print("Early stopping.")
                    break

    # save model weights
    learner.model.save_networks(f"dropout_iter_{index}.pth")

    model_accuracy_test = {}
    for k, dt in test_datasets.items():
        model_accuracy_test[k] = learner.score(dt)[0]
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
        6: loss_weighted_max_entropy,
        7: loss_weighted_bald,
        8: loss_weighted_var_ratios,
        9: loss_weighted_mean_std,
    }
    return acq_func_dict[acq_func]
