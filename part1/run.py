import torch
from utils.train import train
from constants import params
from models.builder import build_classifier
import numpy as np
import random

def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG

def loss_func(y_pred, y):
    return torch.nn.functional.cross_entropy(
        y_pred, y
    )

def metric_funcs(y_pred, y):
    return None

def train_step(x, y, model, optim):
    optim.zero_grad()
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    loss.backward()
    optim.step()
    metrics = metric_funcs(y_pred, y)
    return loss.item(), metrics

def test_step(x, y, model):
    with torch.no_grad():
        y_pred = model(x)
    loss = loss_func(y_pred, y)
    metrics = metric_funcs(y_pred, y)
    return loss.item(), metrics

if __name__ == '__main__':
    set_seeds(params['manual_seed'])
    datadir = 'data'
    logdir = 'logs'
    model = build_classifier(params).to(params['device'])
    done = train(
        params,
        model,    
        train_step,
        test_step,
        datadir,
        logdir
    )
    if done:
        print('Training Done.')
