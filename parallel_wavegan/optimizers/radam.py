import math

import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0):
        # if not 0.0 <= lr:
        #     raise ValueError("Invalid learning rate: {}".format(lr))
        
        # if not 0.0 <= eps:
        #     raise ValueError("Invalid epsilon: {}".format(eps))

        # if not 0.0 <= betas[0] < 1.0:
        #     raise ValueError("Invalid first beta: {}".format(betas[0]))
        # if not 0.0 <= betas[1] < 1.0:
        #     raise ValueError("Invalid second beta: {}".format(betas[1]))
        

        # if isinstance(params,(list,tuple)) and len(params) > 0 and isinstance(params[0],dict):
        #     for param in params:
        #         if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
        #             param['buffer'] = [[None,None,None] for ind in  range(10)]
        defaults = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay) #buffer[i][0] = step ,buffer[i][1] = Simple moving average length, buffer[i][0] = step_size
        self.buffer = [[None,None,None] for ind in range(10)]
        super(RAdam,self).__init__(params,defaults)


    def __setstate__(self, state):
        super(RAdam,self).__setstate__(state)

    def step(self,closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None :
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam doesn't support sparse gradients")

                p_data_fp32 = p.data.float()
                state = self.state[p]

                if(len(state) == 0):
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg,exp_avg_sq = state['exp_avg'],state['exp_avg_sq'] # DW , SW
                beta1,beta2 = group['betas']

                exp_avg.mul_(beta1).add_(1-beta1,grad) # DW = B1*DW+(1-B1)*grad
                exp_avg_sq.mul_(beta2).addcmul_(1-beta2,grad,grad) # SW = B2*Sw + (1-B2)*grad*grad
                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if(state['step'] == buffered[0]):
                    N_sma,step_size = buffered[1],buffered[2] #N_sma = simple moving average length
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    
                    bias_c2 = 1 - beta2_t
                    bias_c1 = 1 - beta1 ** state['step']

                    # calculating N_SMA
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma  = N_sma_max - 2 * state['step'] * beta2_t / bias_c2
                    buffered[1] = N_sma

                    # calculating step_size
                    if N_sma >= 5:
                        # rect_term = math.sqrt(((N_sma-4)*(N_sma-2)*N_sma_max) / ((N_sma_max-4)*(N_sma_max-2)*N_sma))
                        # step_size = math.sqrt(bias_c2) * rect_term / bias_c1
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])  # NOQA


                    # elif self.degenerated_to_sgd:
                    #     step_size = 1.0 / bias_c1
                    else:
                        # step_size = -1
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay']*group['lr'],p_data_fp32)

                if N_sma >= 5: #apply rectified term
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'],exp_avg,denom)
                    
                else: # SGD with momentum 
                    p_data_fp32.add_(-step_size*group['lr'],exp_avg)
    
                p.data.copy_(p_data_fp32)

        return loss


                    



                    





        












                    



