import torch
from data.dataset import get_dataloader
import os
from torch.utils import tensorboard

def train(
        params,
        model,
        train_step,
        test_step,
        datadir,
        logdir
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
        for x, y in train_loader:
            loss, metrics = train_step(x, y, model, optim)
            writer.add_scalar('loss/train', loss)
            if metrics:
                for key, metric in metrics.items():
                    writer.add_scalar('{}/train'.format(key), metric)

        if epoch % params['eval_freq'] == 0:
            count = 0
            for x, y in test_loader:
                loss, metrics = test_step(x, y, model)
                writer.add_scalar('loss/test', loss)
                count += 1
                # Only the first metric determines when lr is decreased with plateau
                avg_val_metric += metrics[0]
                if metrics:
                    for key, metric in metrics.items():
                        writer.add_scalar('{}/test'.format(key), metric)
            assert count > 0
            avg_val_metric = avg_val_metric / count
        
        if epoch % params['save_freq']:
            state_dict = {
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optim.state_dict(),
            }
            for key, item in schedulers.item():
                state_dict[key + 'state_dict'] = item.state_dict()
            torch.save(state_dict, os.path.join(logdir, 'model_epoch_{}.pt'.format(epoch)))

        for key, item in schedulers:
            if key == 'ReduceLROnPlateauSchedule':
                if epoch % params['eval_freq'] == 0:
                    item.step(avg_val_metric)
            else:
                item.step()
    return True
