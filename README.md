# Model Training
The API provides a configurable interface for training a PyTorch Image Classifier.
Any configuration in done using `params`, a python `dict` containing all configurable options of the API.
The following is an example  `params` dict declared in [constants.py](constants.py).
```
params = { 
    'GOOGLE_APPLICATION_CREDENTIALS' : 'neuroengineering-334116-422f8a1aefd9.json',
    'PUB_SUB_TOPIC'                  : 'python-tester',
    'PUB_SUB_PROJECT'                : 'neuroengineering-334116',
    'PUB_SUB_SUBSCRIPTION'           : 'python-tester-sub',
    'TIMEOUT'                        : 3.0,

    'KAFKA_HOST'                     : 'localhost:9092',
    'KAFKA_TOPIC'                    : 'python-tester',
    
    'batch_size'                     : 512,
    'image_height'                   : 28, 
    'image_width'                    : 28, 
    'shuffle_dataset'                : True,
    'flip_images'                    : True,
    'flip_probability'               : 0.4,
    'normalise_images'               : False,
    'manual_seed'                    : 17, 
    'is_image_color'                 : False,
    'normalise_mean'                 : [0.5],
    'normalise_std'                  : [0.25],
    'additional_transforms'          : None,
    'net_arch'                       : [
                                            {
                                                'class' : torch.nn.Conv2d,
                                                'kwargs' : {
                                                    'in_channels' : 1,
                                                    'out_channels' : 32,
                                                    'kernel_size' : 5,
                                                    'stride' : 2,
                                                    'padding' : 0
                                                },
                                            },
                                            {
                                                'class' : torch.nn.BatchNorm2d,
                                                'kwargs' : {
                                                    'num_features' : 32
                                                }
                                            },
                                            {
                                                'class' : torch.nn.ELU,
                                                'kwargs' : {}
                                            },
                                            {
                                                'class' : torch.nn.MaxPool2d,
                                                'kwargs' : {
                                                    'kernel_size' : 2,
                                                    'stride' : 1,
                                                    'padding' : 0,
                                                }
                                            },
                                            {
                                                'class' : torch.nn.Conv2d,
                                                'kwargs' : {
                                                    'in_channels' : 32,
                                                    'out_channels' : 64,
                                                    'kernel_size' : 4,
                                                    'stride' : 1,
                                                    'padding' : 0
                                                },
                                            },
                                            {
                                                'class' : torch.nn.BatchNorm2d,
                                                'kwargs' : {
                                                    'num_features' : 64
                                                }
                                            },
                                            {
                                                'class' : torch.nn.ELU,
                                                'kwargs' : {}
                                            },
                                            {
                                                'class' : torch.nn.MaxPool2d,
                                                'kwargs' : {
                                                    'kernel_size' : 2,
                                                    'stride' : 1,
                                                    'padding' : 0,
                                                }
                                            },
                                            {
                                                'class' : torch.nn.Conv2d,
                                                'kwargs' : {
                                                    'in_channels' : 32,
                                                    'out_channels' : 64,
                                                    'kernel_size' : 4,
                                                    'stride' : 1,
                                                    'padding' : 0
                                                },
                                            },
                                            {
                                                'class' : torch.nn.BatchNorm2d,
                                                'kwargs' : {
                                                    'num_features' : 64
                                                }
                                            },
                                            {
                                                'class' : torch.nn.ELU,
                                                'kwargs' : {}
                                            },
                                            {
                                                'class' : torch.nn.MaxPool2d,
                                                'kwargs' : {
                                                    'kernel_size' : 2,
                                                    'stride' : 1,
                                                    'padding' : 0,
                                                }
                                            },
                                            {
                                                'class' : torch.nn.Conv2d,
                                                'kwargs' : {
                                                    'in_channels' : 64,
                                                    'out_channels' : 128,
                                                    'kernel_size' : 3,
                                                    'stride' : 1,
                                                    'padding' : 0
                                                },
                                            },
                                            {
                                                'class' : torch.nn.BatchNorm2d,
                                                'kwargs' : {
                                                    'num_features' : 128
                                                }
                                            },
                                            {
                                                'class' : torch.nn.ELU,
                                                'kwargs' : {}
                                            },
                                            {
                                                'class' : torch.nn.MaxPool2d,
                                                'kwargs' : {
                                                    'kernel_size' : 2,
                                                    'stride' : 1,
                                                    'padding' : 0,
                                                }
                                            },
                                            {
                                                'class' : torch.nn.Conv2d,
                                                'kwargs' : {
                                                    'in_channels' : 128,
                                                    'out_channels' : 256,
                                                    'kernel_size' : 3,
                                                    'stride' : 1,
                                                    'padding' : 0
                                                },
                                            },
                                            {
                                                'class' : torch.nn.BatchNorm2d,
                                                'kwargs' : {
                                                    'num_features' : 256
                                                }
                                            },
                                            {
                                                'class' : torch.nn.ELU,
                                                'kwargs' : {}
                                            },
                                            {
                                                'class' : torch.nn.MaxPool2d,
                                                'kwargs' : {
                                                    'kernel_size' : 2,
                                                    'stride' : 1,
                                                    'padding' : 0,
                                                }
                                            }
                                        ],
    'n_classes'                      : 10,
    'num_epochs'                     : 1000,
    'eval_freq'                      : 5,
    'save_freq'                      : 5,
    'optim_class'                    : torch.optim.SGD,
    'optim_kwargs'                   : {
                                            # All kwargs other than learning rate here
                                            'momentum' : 0,
                                            'dampening' : 0,
                                            'weight_decay' : 1e-2,
                                            'nesterov' : False,
                                        },
    'learning_rate'                  : 1e-1,
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
                                                    'threshold' : 1e-5,
                                                }
                                            }
                                        ],
    'device'                         : 'cuda' if torch.cuda.is_available() else 'cpu',
    'mode'                           : 'notebook',
    'eval_metric_compute_freq'       : 'epoch',
    'eval_batch_size'                : 1,
}
```
The keys are named so as to signify their meaning in standard ML terminology.
Please modify this config file to change training configurations.
The following are the declared keys for Google Pub/Sub config:
- `GOOGLE_APPLICATION_CREDENTIALS`: name of config.json for Google Cloud Compute Service Account for Google Pub/Sub Access
- `PUB_SUB_TOPIC` : Google Pub/Sub topic to publish paylod to. The topic must be already created, else an error will be raised
- `PUB_SUB_PROJECT` : Project Name for service account used
- `PUB_SUB_SUBSCRIPTION` : Subscription name to subscribe for pulling messages from Google Pub/Sub. The subscription must be created with the topic.
- `TIMEOUT` : Timeout for Google Pub/Sub
The following are the declared keys for Apache Kafka config:
- `KAFKA_HOST` : Host for Apache Kafka Broker. Defaut port: 9092 used
- `KAFKA_TOPIC` : Kafka Topic to publish to and subscribe from
The following are the declared keys for Deep Learning config:
- `batch_size`: Batch Size to be used during training
- `image_height` : Image Height input to the model
- `image_width` : Image Width input to the model
