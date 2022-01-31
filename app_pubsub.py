import torch
import numpy as np
import os
from pubsub import publish_payload, consume_payload
from models.builder import build_classifier
from constants import params
from data.dataset import get_dataloader
import time
from google.cloud import pubsub_v1 as ps
import json
import cv2

if __name__ == '__main__':
    model_name = 'model_epoch_995.pt'
    params['batch_size'] = params['eval_batch_size']
    test_loader = get_dataloader(params, 'test')
    model = build_classifier(params).to(params['device'])
    state_dict = torch.load(
        os.path.join('logs', model_name),
        map_location = torch.device(params['device'])
    )
    model.load_state_dict(state_dict['model_state_dict'])
    #model.eval()

    def process_payload(message):
        message.ack()
        message = json.loads(message.data)
        #print("Received {}.".format(message))
        img = np.array(message['image'], dtype = np.float32)
        x = torch.from_numpy(
            (np.expand_dims(
                np.expand_dims(img, 0), 0
            ) / 255.0).astype(np.float32)
        )
        y_pred = model(x)
        print('Processed Output: {}, Target: {}'.format(torch.argmax(y_pred, -1).detach().cpu().numpy()[0], message['target']))

    publisher = ps.PublisherClient()
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(params['device']), y.to(params['device'])
        img = x.detach().cpu().numpy()[0][0]
        y = int(y.detach().cpu().numpy()[0])
        payload = {"data" : 'Payload Data', "timestamp": time.time(), "image" : img.tolist(), "target" : y}
        publish_payload(params, publisher, payload, params['PUB_SUB_TOPIC'])
        subscriber = ps.SubscriberClient()
        consume_payload(params, params['PUB_SUB_SUBSCRIPTION'], process_payload, subscriber)
