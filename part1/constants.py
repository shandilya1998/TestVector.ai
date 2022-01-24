import torch

params = {
    'batch_size'                     : 512,
    'image_height'                   : 28,
    'image_width'                    : 28,
    'shuffle_dataset'                : True,
    'rotate_images'                  : True,
    'normalise_images'               : True,
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
}
