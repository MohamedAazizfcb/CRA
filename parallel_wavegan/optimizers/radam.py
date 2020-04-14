import math

import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0,degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid first beta: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid second beta: {}".format(betas[1]))
        self.degenerated_to_sgd = degenerated_to_sgd

        if isinstance(params,(list,tuple)) and len(params) > 0 and isinstance(params[0],dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None,None,None] for ind in  range(10)]
        defaults = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,buffer=[[None,None,None] for ind in range(10)]) #buffer[i][0] = step ,buffer[i][1] = Simple moving average length, buffer[i][0] = step_size
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

                    p_data_f = p.data.float()
                    state = self.state[p]

                    if(len(state) == 0):
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p_data_f)
                        state['exp_avg_sq'] = torch.zeros_like(p_data_f)
                    else:
                        state['exp_avg'] = state['exp_avg'].type_as(p_data_f)
                        state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_f)

                    exp_avg,exp_avg_sq = state['exp_avg'],state['exp_avg_sq'] # DW , SW
                    beta1,beta2 = group['betas']

                    exp_avg.mul_(beta1).add_(1-beta1,grad) # DW = B1*DW+(1-B1)*grad
                    exp_avg_sq.mul_(beta2).addcmul_(1-beta2,grad,grad) # SW = B2*Sw + (1-B2)*grad*grad
                    state['step'] += 1
                    buffered = group['buffer'][int(state['step'] % 10)]

                    if(state['step'] == buffered[0]):
                        sma_len,step_size = buffered[1],buffered[2] #sma_len = simple moving average length
                    else:
                        buffered[0] = state['step']

                        beta2_pow_t = beta2 ** state['step']
                        bias_c2 = 1 - beta2_pow_t
                        bias_c1 = 1 - beta1 ** state['step']

                        # calculating SMA_len
                        sma_max_len = 2 / (1 - beta2) - 1
                        sma_len  = sma_max_len - 2 * state['step'] * beta2_pow_t / bias_c2
                        buffered[1] = sma_len

                        # calculating step_size
                        if sma_len >= 5:
                            rect_term = math.sqrt(((sma_len-4)*(sma_len-2)*sma_max_len) / ((sma_max_len-4)*(sma_max_len-2)*sma_len))
                            step_size = math.sqrt(bias_c2) * rect_term / bias_c1
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / bias_c1
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if group['weight_decay'] != 0:
                        p_data_f.add_(-group['weight_decay']*group['lr'],p_data_f)

                    if sma_len >= 5: #apply rectified term
                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                        p_data_f.addcdiv_(-step_size * group['lr'],exp_avg,denom)
                        
                    elif step_size > 0: # SGD with momentum 
                        p_data_f.add_(-step_size*group['lr'],exp_avg)
        
                    p.data.copy_(p_data_f)

            return loss


                    



                    





        












                    



