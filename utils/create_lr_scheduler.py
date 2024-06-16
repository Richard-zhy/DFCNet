'''
用于创建学习率动态调整策略
'''

import  torch
def create_lr_scheduler(optimizer, num_step:int, epochs:int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step>0 and epochs>0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x<= warmup_epochs:
            alpha = float(x) / warmup_epochs
            return warmup_factor * (1-alpha) + alpha
        else:
            return (1 - (x - warmup_epochs) / (epochs - warmup_epochs))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


