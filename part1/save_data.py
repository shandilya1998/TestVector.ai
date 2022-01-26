import cv2
from utils import mnist_reader
import pandas as pd
import os
from tqdm import tqdm


if __name__ == '__main__':
    train_x, train_y = mnist_reader.load_mnist('data/train', 'train')
    train_x = train_x.reshape((train_x.shape[0], 28, 28))
    test_x, test_y = mnist_reader.load_mnist('data/test', 'test')
    test_x = test_x.reshape((test_x.shape[0], 28, 28))
    train_data = {'files' : [], 'type': []}
    test_data = {'files': [], 'type' : []}
    for i in tqdm(range(train_x.shape[0])):
        f = 'FashionMNIST_{}.png'.format(i)
        cv2.imwrite(os.path.join('data', 'train', f), train_x[i])
        train_data['files'].append(f)
        train_data['type'].append(train_y[i])

    train = pd.DataFrame(train_data)
    train.to_csv(
        os.path.join('data', 'train', 'info.csv'), index = False
    )

    for j in tqdm(range(test_x.shape[0])):
        f = 'FashionMNIST_{}.png'.format(j)
        cv2.imwrite(os.path.join('data', 'test', f), test_x[j])
        test_data['files'].append(f)
        test_data['type'].append(test_y[j])

    test = pd.DataFrame(test_data)
    test.to_csv(
        os.path.join('data', 'test', 'info.csv'),
        index = False
    )

