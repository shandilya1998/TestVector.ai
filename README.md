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
- `shuffle_dataset` : Boolean to shuffle dataset at epoch end
- `flip_images` : Boolean to apply random horizontal and vertical flip
- `flip_probability` : Probability of applying random horizontal and vertical flip
- `normalise_images` : Boolean to normalise data points
- `manual_seed` : Manual Seed for pseudo random number generators for numpy and torch
- `is_image_color` : Boolean, True if input images are color. Default is False. API not tested for color images but mechanism is provided
- `normalise_mean` : List of data normalisation tranform mean. Length one list for grayscale images, 3 for color images
- `normalise_std` : List of data normalisation tranform standard deviation. Length one list for grayscale images, 3 for color images
- `additional_transforms` : List of additional transforms to be applied to the data. Default value None if no additional transforms needed
- `net_arch` : List of Dict containing classifier declarationg
- `n_classes` : Number of classes for the classification problem
- `num_epochs` : Number of training epochs
- `eval_freq` : Evaluation done after every `params['eval_freq']` epochs
- `save_freq` : Model saved after very `params['save_freq']` epochs
- `optim_class` : Optimiser class used for training
- `optim_kwargs` : Arguments other than learning rate needed to initialise optimiser
- `learning_rate` : Learning Rate for Deep Learning
- `learning_rate_schedule` : List of Dicts of Pytorch Learning Rate Scheduler class and arguments needed for initialisation
- `device` : `cuda` or `cpu`. Default Programmatically decided
- `eval_metric_compute_freq` : Compute Evaluation Metrics every `params['eval_metric_compute_freq']`. Default `epoch`.
- `eval_batch_size` : Evaluation Batch Size. Default `1`

## Train API

Model Training is facilitated through the `def train(**kwargs)` defined in [utils/train.py](utils/train.py)
The method signature is as follows:
```
def train(
        params,
        model,
        train_step,
        test_step,
        datadir,
        logdir,
        **kwargs
    ) -> bool
```

The following are the arguments required by `def train(**kwargs)`
- `params` contains the configuration for all the training components such as the optimiser, learning rate schedulers and metrics.
- `model` is the `torch.nn.Module` of the classifier
- `train_step` is a method that takes the inputs and targets to compute and apply gradients
- `test_step` is a method that takes the validation input and targets to compute evaluation loss and metrics
- `datadir` is the relative path to the directory storing the data
- `logdir` is the directory used to store `Tensorboard` logs and saved model files
- `**kwargs` are the additional arguments such as `train_torchmetrics` and `test_torchmetrics`

### Train Step
`def train_step(**kwargs)` is passed to `def train(**kwargs)` and is called to compute and apply gradients.
This method needs to be defined by the user.
The following is a sample `def train_step(**kwargs)`:

```
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
```

Any Implementation must return the float loss and dict of float metrics
### Test Step
`def test_step(**kwargs)` is passed to `def train(**kwargs)` and is called to compute validation metrics and loss.
This method needs to be defined by the user.
The following is a sample `def train_step(**kwargs)`:

```
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
```

Any Implementation must return the float loss and dict of float metrics

### Loss Function
`def loss_func(y_pred, y)` must be defined in the same scope as `def train_step(**kwargs)` and `def test_step(**kwargs)`, and must be available to be called within these methods.
The following is a sample implementation using Cross-Entropy loss:

```
def loss_func(y_pred, y):
    return torch.nn.functional.cross_entropy(
        y_pred, y
    )
```


### Model Building
The API provides a factory function for creating a classifier `torch.nn.Module`.
Model Builder is declared in [models/builder.py](models/builder.py).
The following is the method signature:

```
def build_classifier(params) -> torch.nn.Module
```

The following items in `params` are used in the method and must be defined:
- `image_height`
- `image_width`
- `is_image_color`
- `net_arch`

The method appends a Fully Connected Layer followed my Softmax Activation so that the network outputs classification probabilities.

### Dataset API

The image classification pipeline can read grayscale and color images and train a classification model appropriately
The training images must be stored in the folder [data/train](data/train).
The target values are stored in [data/train/info.csv](data/train/info.csv) with two columns: `files` and `type` where `files` is the image file name and `type` is the classification label.

The FashionMNIST dataset is available as a zip file containing a numpy array of all images and labels in 
`train-images-idx3-ubyte.gz`, 
`train-labels-idx1-ubyte.gz`, 
`test-labels-idx1-ubyte.gz` and 
`test-images-idx3-ubyte.gz`. 
If the data is similar to aforementioned format, run the python script [save_data.py](save_data.py) to unzip and store data appropriately for pipeline to work.

#### FashionMNIST Dataset
The API provides a simple interface for loading the data in [data](data). 
`class FashionMNIST` inherits from `torch.utils.data.Dataset` and takes the following two arguments during initialisation:
- `params`
- 'kind': test or train identifier.

#### Dataloader Builder
The FashionMNIST dataset can be directly loaded into a dataloader for training.
The following is the function signature:
```
get_dataloader(params, kind) -> torch.utils.data.DataLoader
```

### Sample Use Case

```
set_seeds(params['manual_seed'])
datadir = 'data'
logdir = 'logs'
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
    ]).to(params['device'])
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

```

# Streaming API
The Streaming API provides methods for interfacing with Google Pub\Sub and Apache Kafka.
The API builds on top of the python packages `google-cloud-pubsub` and `kafka-python`. 
Google Cloud SDK and Apache Kafka binaries must be available to run the Streaming API.

## Google Pub/Sub
Streaming API endpoints for Google Pub/Sub are declared in [pubsub.py](pubsub.py).
The following methods are available:
- `def publish_payload(params, publisher, payload, topic, **kwargs)`
- `def consume_payload(params, subscription, callback, subscriber, **kwargs)`

Provision for additional arguments for the `google-cloud-pubsub` package  methods is provided through `**kwargs`.

The following sample depicts usage of the aforementioned methods:
```
publisher = ps.PublisherClient()
step = 0 
img = cv2.imread(os.path.join('assets', 'FashionMNIST_0.png'), cv2.IMREAD_GRAYSCALE)
while step < 5:
    print("===================================")
    payload = {"data" : 'Payload Data', "timestamp": time.time(), "image" : img.tolist()}
    print(f"Sending payload: {payload}.")
    publish_payload(params, publisher, payload, params['PUB_SUB_TOPIC'])
    subscriber = ps.SubscriberClient()
    consume_payload(params, params['PUB_SUB_SUBSCRIPTION'], process_payload, subscriber)
    step += 1
```

`def process_payload(message)` must be defined by the user to process the received message and must contain a 'message.ack()' call followed by the message processing statements.
The following is a sample implementation:
```
def process_payload(message):
    message.ack()
    message = json.loads(message.data)
    print("Received {}.".format(message))
```

Run [pubsub.py](pubsub.py) to view streaming demo.

## Apache Kafka
Streaming API endpoints for Apache Kafka are declared in [kf.py](kf.py).
The following methods are available:
- `def publish_payload(params, publisher, payload, topic, **kwargs)`
- `def consume_payload(params, consumer, topic, callback, **kwargs)`

`def process_payload(message)` must be defined by the user to process the received message.
The following is a sample implementation for usage of the aforementioned methods:
```
import json
import time
import os
from constants import params
import threading
import cv2 
import numpy as np

def process_payload(message):
    print("===================================")
    print("Received")
    print("===================================")
    payload = json.loads(message.value)
    img = np.array(payload['image'], dtype = np.uint8)
    cv2.imshow('decoded', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass

def start_producing():
    publisher = kf.KafkaProducer(bootstrap_servers = params['KAFKA_HOST'])
    step = 0 
    img = cv2.imread(os.path.join('assets', 'FashionMNIST_0.png'), cv2.IMREAD_GRAYSCALE)
    while True:
        print("===================================")
        #cv2.imshow('original', img)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    pass
        payload = {"data" : "Payload data", "image" : img.tolist(), "timestamp": time.time()}
        print("Sending payload")
        print("===================================")
        publish_payload(params, publisher, payload, params['KAFKA_TOPIC'])
        step += 1
    
def start_consuming():
    while True:
        consumer = kf.KafkaConsumer(params['KAFKA_TOPIC'], bootstrap_servers = params['KAFKA_HOST'])
        consume_payload(params, consumer, params['KAFKA_TOPIC'], process_payload)

if __name__ == '__main__':
    threads = []
    t = threading.Thread(target=start_producing)
    t2 = threading.Thread(target=start_consuming)
    threads.append(t)
    threads.append(t2)
    t.start()
    t2.start()
```

Run [kf.py](kf.py) for streaming demo.
[start_kafka.sh](start_kafka.sh) must be executed and topics must be created using Apache Kafka console scripts before running the script.



