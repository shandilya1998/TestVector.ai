import torch
from utils.train import train
from constants import params
from models.builder import build_classifier

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
    datadir = 'data'
    logdir = 'logs'
    model = build_classifier(params)
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
