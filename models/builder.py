import torch

def build_classifier(params):
    layers = []
    num_channels = 1
    if params['is_image_color']:
        num_channels = 3
    inp = torch.zeros((1, num_channels, params['image_height'], params['image_width']))
    out = None
    for layer in params['net_arch']:
        # params['net_arch'] only defines the convolutional pipeline
        layers.append(layer['class'](**layer['kwargs']))
        with torch.no_grad():
            out = layers[-1](inp)
        inp = out
    # classifying fully connected layer defined here
    layers.append(torch.nn.Flatten())
    out = layers[-1](inp)
    n_flatten = out.shape[-1]
    layers.append(torch.nn.Linear(
        n_flatten,
        params['n_classes']
    ))
    layers.append(torch.nn.Softmax(dim = -1))
    return torch.nn.Sequential(*layers)
