import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys, os
sys.path.append(os.path.abspath(os.path.join('..','CS545-Real-Fake-Image-Detection')))
from models import get_model
from active_learning.active_learning_loss import BCEWithLogitsLoss as DynamicWeightsBCE

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = get_model(opt.arch)

        if opt.ckpt is not None:
            print(f"Loading Checkpoint from {opt.ckpt}")
            state_dict = torch.load(opt.ckpt, map_location='cpu')
            if "model" in list(state_dict.keys()):
                state_dict = state_dict["model"]
                state_dict = {"weight":state_dict["fc.weight"],
                                "bias":state_dict["fc.bias"]}
            self.model.fc.load_state_dict(state_dict)
        else:
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = DynamicWeightsBCE() if opt.use_weighted_loss else nn.BCEWithLogitsLoss()

        # set current device and transfer model to it
        if len(opt.gpu_ids) > 1:
            self.device = torch.cuda.current_device() 
        elif len(opt.gpu_ids) == 1:
            self.device = opt.gpu_ids[0] 
        else: 
            self.device = "cpu"
        self.model.to(self.device)

        # enable distributed parallel processing if num gpus > 1
        if len(opt.gpu_ids) > 1:
            # Make model replica operate on the current device
            self.model = torch.nn.parallel.DistributedDataParallel(
                module=self.model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=False
            )


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)

    
    def forward_raw(self):
        return self.model(self.input)


    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self, weights: torch.Tensor=None):
        self.forward()
        if self.opt.use_weighted_loss:
            assert weights is not None, "Error: given type None as weights"
            self.loss = self.loss_fn(self.output.squeeze(1), 
                                    self.label,
                                    weights) 
        else:
            self.loss = self.loss_fn(self.output.squeeze(1), 
                                    self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



