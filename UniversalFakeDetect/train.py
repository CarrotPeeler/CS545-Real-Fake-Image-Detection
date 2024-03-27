import os
import time
import numpy as np
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader, shuffle_dataset
from data.datasets import RealFakeDataset
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
from tqdm.auto import tqdm
import torch
import distributed as du
import numpy as np
from skorch import NeuralNetClassifier
from active_learning.active_learning_net import active_learning_procedure, select_acq_function 


def load_al_model(model, opt):
    """Scikit-Learn wrapper for PyTorch models"""
    estimator = NeuralNetClassifier(
        module=model.model,
        lr=opt.lr,
        batch_size=opt.batch_size,
        max_epochs=opt.niter,
        criterion=model.loss_fn,
        optimizer=model.optimizer,
        train_split=None,
        verbose=0,
        device=model.device,
    )
    return estimator


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    # num samples to validate for each real and fake val set (14k total)
    val_opt.max_sample = 1000 if opt.uniform_sample else 7000 
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


def train(model, opt, val_opt):
    if len(opt.gpu_ids) > 1:
        du.init_distributed_training(len(opt.gpu_ids), opt.shard_id)
    
    # Set random seed from configs.
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    model = Trainer(opt)
    
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)
    print(f"NUM TRAIN SAMPLES: {len(data_loader.dataset)}\nNUM VAL SAMPLES: {len(val_loader.dataset)}")

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    # run training epochs
    for epoch in tqdm(range(opt.niter)):
        # set current epoch for the data loader
        if len(opt.gpu_ids) > 1:
            shuffle_dataset(data_loader, epoch)
            if hasattr(data_loader.dataset, "_set_epoch_num"):
                data_loader.dataset._set_epoch_num(epoch)

        # perform mini-batch training
        for i, data in enumerate(tqdm(data_loader)):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )

            if model.total_steps in [10,30,50,100,1000,5000,10000] and False: # save models at these iters 
                model.save_networks('model_iters_%s.pth' % model.total_steps)
            # sync GPUs
            torch.cuda.synchronize()

        # create model checkpoint
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( 'model_epoch_best.pth' )
            model.save_networks( 'model_epoch_%s.pth' % epoch )

            # Validation
            model.eval()
            ap, r_acc, f_acc, acc = validate(model.model, val_loader, gpu_id=model.device)
            if len(opt.gpu_ids) > 1:
                ap, acc = du.all_reduce([ap, acc])
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

            early_stopping(acc, model)
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training...")
                    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    print("Early stopping.")
                    break
        model.train()
        # in case of fragmented memory
        torch.cuda.empty_cache()


def train_active_learning(opt, val_opt):
    # Set random seed from configs.
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    model = Trainer(opt) # load trainer
    estimator = load_al_model(model, opt) # wrap trainer 

    # create init_train, val, and pool datasets

    # fake/real train samples are both ~240k, sample random indices for init
    init_idxs = np.random.choice(240000, opt.num_samples_per_class) 
    init_loader = create_dataloader(opt, premade_dataset=RealFakeDataset(opt, "init", init_idxs))
    pool_loader = create_dataloader(opt, premade_dataset=RealFakeDataset(opt, "pool", init_idxs))
    val_loader = create_dataloader(val_opt)

    print(f"NUM INIT SAMPLES: {len(init_loader.dataset)}\
          \nNUM POOL SAMPLES: {len(pool_loader.dataset)}\
          \nNUM VAL SAMPLES: {len(val_loader.dataset)}")


    # # Wrap train/val dataloaders for Active Learning
    # datasets = dict()
    # dataloader = LoadData(train_loader, 
    #                       val_loader, 
    #                       num_samples_per_class=opt.num_samples_per_class)
    # (
    #     datasets["X_init"],
    #     datasets["y_init"],
    #     datasets["X_val"],
    #     datasets["y_val"],
    #     datasets["X_pool"],
    #     datasets["y_pool"]
    # ) = dataloader.load_all()

    # acq_func = select_acq_function(opt.acq_func)
    # training_hist, test_score = active_learning_procedure(
    #                                 query_strategy=acq_func,
    #                                 X_val=datasets["X_val"],
    #                                 y_val=datasets["y_val"],
    #                                 X_pool=datasets["X_pool"],
    #                                 y_pool=datasets["y_pool"],
    #                                 X_init=datasets["X_init"],
    #                                 y_init=datasets["y_init"],
    #                                 estimator=estimator,
    #                                 T=args.dropout_iter,
    #                                 n_query=args.query,
    #                                 training=state,
    #                             )

if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()

    if opt.use_active_learning:
        train_active_learning(opt, val_opt) # cannot run on more than 1 GPU currently
    else:
        du.launch_job(opt=opt, val_opt=val_opt, init_method=opt.init_method, func=train)
 
    
        
   

