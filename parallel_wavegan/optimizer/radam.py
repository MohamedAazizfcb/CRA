import math
import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0):
        defaults = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        self.buffer = [[None,None,None] for ind in range(10)]
        super(RAdam,self).__init__(params,defaults)
