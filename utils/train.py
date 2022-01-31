import torch
from data.dataset import get_dataloader
import os
from torch.utils import tensorboard
from constants import params

def train(
        params,
        model,
        train_step,
        test_step,
        datadir,
        logdir,
        **kwargs
    ):
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
    for epoch in range(params['num_epochs']):
        avg_val_metric = 0.0
        avg_epoch_loss = 0.0
        is_metric_available = False
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(params['device']), y.to(params['device'])
            loss, metrics = train_step(x, y, model, optim, **kwargs)
            avg_epoch_loss += loss
            writer.add_scalar('loss/train', loss, epoch * len(train_loader) + i)
            if metrics:
                for key, metric in metrics.items():
                    writer.add_scalar('{}/train'.format(key), metric, epoch * len(train_loader) + i)
        avg_epoch_loss = avg_epoch_loss / len(train_loader)
        print('Epoch {} done. Average Epoch Loss {:.8f}'.format(epoch, avg_epoch_loss))
        writer.add_scalar('avg_loss/train', avg_epoch_loss, epoch)
        if epoch % params['eval_freq'] == 0:
            count = 0
            avg_val_epoch_loss = 0.0
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(params['device']), y.to(params['device'])
                loss, metrics = test_step(x, y, model, **kwargs)
                avg_val_epoch_loss += loss
                writer.add_scalar('loss/test', loss, (epoch % params['eval_freq']) * len(test_loader) + i)
                count += 1
                # Only the first metric determines when lr is decreased with plateau
                if metrics:
                    is_metric_available = True
                    avg_val_metric += metrics[0]
                    for key, metric in metrics.items():
                        writer.add_scalar('{}/test'.format(key), metric, (epoch % params['eval_freq']) * len(test_loader) + i)
            assert count > 0
            avg_val_epoch_loss = avg_val_epoch_loss / count
            print('----------------------------------------------')
            print('Evaluation {} done. Average Loss {:.8f}'.format(int(epoch / params['eval_freq']),  avg_val_epoch_loss))
            if params['eval_metric_compute_freq'] == 'epoch' and 'test_torchmetrics' in kwargs.keys():
                metrics = kwargs['test_torchmetrics'].compute()
                print(' '.join(['{} {:.8f}'.format(key, item.item()) for key, item in metrics.items()]))
            print('----------------------------------------------')
            avg_val_metric = avg_val_metric / count
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
    return True
