import torch

params = {
    'batch_size'                     : 512,
    'image_height'                   : 28,
    'image_width'                    : 28,
    'shuffle_dataset'                : True,
    'rotate_images'                  : True,
    'normalise_images'               : False,
    'manual_seed'                    : 117,
    'is_image_color'                 : False,
    'normalise_mean'                 : [0.5],
    'normalise_std'                  : [0.25],
    'additional_transforms'          : None,
    'net_arch'                       : [
                                            {
                                                'class': torch.nn.Conv2d,
                                                'kwargs' : {
                                                    'in_channels' : 1,
                                                    'out_channels' : 32,
                                                    'kernel_size' : 5,
                                                    'stride' : 1,
                                                    'padding' : 0,
                                                }
                                            },
                                            {
                                                'class' : torch.nn.ReLU,
                                                'kwargs' : {}
                                            },
                                            {
                                                'class': torch.nn.Conv2d,
                                                'kwargs' : {
                                                    'in_channels' : 32,
                                                    'out_channels' : 64,
                                                    'kernel_size' : 3,
                                                    'stride' : 2,
                                                    'padding' : 0,
                                                }
                                            },
                                            {
                                                'class' : torch.nn.ReLU,
                                                'kwargs' : {}
                                            },
                                            {
                                                'class': torch.nn.Conv2d,
                                                'kwargs' : {
                                                    'in_channels' : 64,
                                                    'out_channels' : 32,
                                                    'kernel_size' : 3,
                                                    'stride' : 1,
                                                    'padding' : 0,
                                                }
                                            },
                                            {
                                                'class' : torch.nn.ReLU,
                                                'kwargs' : {}
                                            },
                                            {
                                                'class': torch.nn.Conv2d,
                                                'kwargs' : {
                                                    'in_channels' : 32,
                                                    'out_channels' : 8,
                                                    'kernel_size' : 3,
                                                    'stride' : 1,
                                                    'padding' : 0,
                                                }
                                            },
                                            {   
                                                'class' : torch.nn.ReLU,
                                                'kwargs' : {}
                                            },
                                        ],
    'n_classes'                      : 10,
    'num_epochs'                     : 10,
    'eval_freq'                      : 5,
    'save_freq'                      : 5,
    'optim_class'                    : torch.optim.SGD,
    'optim_kwargs'                   : {
                                            # All kwargs other than learning rate here
                                            'momentum' : 0,
                                            'dampening' : 0,
                                            'weight_decay' : 1e-2,
                                            'nesterov' : False
                                        },
    'learning_rate'                  : 1e-3,
    'learning_rate_schedule'         : [
                                            {
                                                'name' : 'ExponentialLRSchedule',
                                                'class' : torch.optim.lr_scheduler.ExponentialLR,
                                                'kwargs' : {
                                                    'gamma' : 0.99,
                                                    'last_epoch' : - 1,
                                                    'verbose' : False
                                                }
                                            }, {
                                                'name' : 'ReduceLROnPlateauSchedule',
                                                'class' : torch.optim.lr_scheduler.ReduceLROnPlateau,
                                                'kwargs' : {
                                                    'mode' : 'min',
                                                    'factor' : 0.5,
                                                    'patience' : 10,
                                                    'threshold' : 1e-3,
                                                }
                                            }
                                        ],
    'device'                         : 'cuda' if torch.cuda.is_available() else 'cpu'
}
