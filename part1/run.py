import torch
from utils.train import train
from constants import params
from models.builder import build_classifier
import numpy as np
import random
import torchmetrics

def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG

def loss_func(y_pred, y):
    return torch.nn.functional.cross_entropy(
        y_pred, y
    )

def train_step(x, y, model, optim, **kwargs):
    optim.zero_grad()
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    loss.backward()
    optim.step()
    metrics = None
    if 'train_torchmetrics' in kwargs.keys():
        metrics = kwargs['train_torchmetrics'](y_pred, y)
        metrics = {key : item.item() for key, item in metrics.items()}
    return loss.item(), metrics

def test_step(x, y, model, **kwargs):
    with torch.no_grad():
        y_pred = model(x)
    loss = loss_func(y_pred, y)
    metrics = None
    if 'test_torchmetrics' in kwargs.keys():
        metrics = kwargs['test_torchmetrics'](y_pred, y)
        if params['eval_metric_compute_freq'] == 'epoch':
            metrics = None
        else:
            metrics = {key : item.item() for key, item in metrics.items()}
    return loss.item(), metrics

if __name__ == '__main__':
    set_seeds(params['manual_seed'])
    datadir = 'data'
    #logdir = 'logs'
    logdir = '/content/drive/MyDrive/Vector/exp1'
    model = build_classifier(params).to(params['device']) 
    kwargs = {
        'train_torchmetrics' : torchmetrics.MetricCollection([
            torchmetrics.Accuracy(num_classes = params['n_classes'], average = 'macro'),
            torchmetrics.Precision(params['n_classes'], average = 'macro'),
            torchmetrics.Recall(params['n_classes'], average = 'macro'),
            torchmetrics.F1Score(params['n_classes'], average = 'macro')
        ]).to(params['device']),
        'test_torchmetrics' : torchmetrics.MetricCollection([
            torchmetrics.Accuracy(num_classes = params['n_classes'], compute_on_step = False, average = 'macro'),
            torchmetrics.Precision(params['n_classes'], compute_on_step = False, average = 'macro'),
            torchmetrics.Recall(params['n_classes'], compute_on_step = False, average = 'macro'),
            torchmetrics.F1Score(params['n_classes'], compute_on_step = False, average = 'macro')
        ].to(params['device']))
    }

    done = train(
        params,
        model,    
        train_step,
        test_step,
        datadir,
        logdir,
        **kwargs
    )
    if done:
        print('Training Done.')
