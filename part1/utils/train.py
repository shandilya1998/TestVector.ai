import torch
from data.dataset import get_dataloader
import os
from torch.utils import tensorboard
from constants import params
if params['mode'] == 'notebook':
    from tqdm import tqdm_notebooks as tqdm
else:
    from tqdm import tqdm
def train(
        params,
        model,
        train_step,
        test_step,
        datadir,
        logdir
    ):
    if params['mode'] == 'notebook':
        from tqdm.notebook import tqdm
    train_loader = get_dataloader(params, 'train')
    test_loader = get_dataloader(params, 'test')
    writer = tensorboard.SummaryWriter(
        log_dir = logdir
    )
    optim = params['optim_class'](
        model.parameters(),
        params['learning_rate'],
        **params['optim_kwargs']
    )
    schedulers = {
        scheduler['name'] :  scheduler['class'](optim, **scheduler['kwargs']) \
            for scheduler in params['learning_rate_schedule']
    }
    model_path = os.path.join(logdir, 'model.pt')
    epoch_bar = tqdm(total = params['num_epochs'], position = 0)
    train_bar = tqdm(total = len(train_loader), position = 1)
    test_bar = tqdm(total = len(test_loader), position = 2)
    for epoch in range(params['num_epochs']):
        avg_val_metric = 0.0
        is_metric_available = False
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(params['device']), y.to(params['device'])
            loss, metrics = train_step(x, y, model, optim)
            writer.add_scalar('loss/train', loss, epoch * len(train_loader) + i)
            train_bar.update(1)
            if metrics:
                for key, metric in metrics.items():
                    writer.add_scalar('{}/train'.format(key), metric, epoch * len(train_loader) + i)
        if epoch % params['eval_freq'] == 0:
            count = 0
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(params['device']), y.to(params['device'])
                loss, metrics = test_step(x, y, model)
                writer.add_scalar('loss/test', loss, (epoch % params['eval_freq']) * len(test_loader) + i)
                count += 1
                test_bar.update(1)
                # Only the first metric determines when lr is decreased with plateau
                if metrics:
                    is_metric_available = True
                    avg_val_metric += metrics[0]
                    for key, metric in metrics.items():
                        writer.add_scalar('{}/test'.format(key), metric, (epoch % params['eval_freq']) * len(test_loader) + i)
            assert count > 0
            avg_val_metric = avg_val_metric / count
            test_bar.refresh()
            test_bar.reset()
        train_bar.refresh()
        train_bar.reset()
        if epoch % params['save_freq'] == 0:
            state_dict = {
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optim.state_dict(),
            }
            for key, item in schedulers.items():
                state_dict[key + 'state_dict'] = item.state_dict()
            torch.save(state_dict, os.path.join(logdir, 'model_epoch_{}.pt'.format(epoch)))

        for key, item in schedulers.items():
            if key == 'ReduceLROnPlateauSchedule':
                if is_metric_available:
                    item.step(avg_val_metric)
            else:
                item.step()
        epoch_bar.update(1)
    epoch_bar.refresh()
    epoch_bar.reset()
    epoch_bar.close()
    train_bar.close()
    test_bar.close()
    return True
