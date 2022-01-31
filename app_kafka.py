import torch
import numpy as np
import os
from kf import publish_payload, consume_payload
from models.builder import build_classifier
from constants import params
from data.dataset import get_dataloader
import time
from google.cloud import pubsub_v1 as ps
import json
import cv2
import threading
import kafka as kf

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

    def process_payload(message):
        message = json.loads(message.value)
        #print("Received {}.".format(message))
        img = np.array(message['image'], dtype = np.float32)
        cv2.imshow('decoded', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
        x = torch.from_numpy(
            (np.expand_dims(
                np.expand_dims(img, 0), 0
            ) / 255.0).astype(np.float32)
        )
        y_pred = model(x)
        print('Processed Output: {} Target Value {}'.format(torch.argmax(y_pred, -1).detach().cpu().numpy()[0], message['target']))

    def start_producing():
        """
            Multiple Publishers can be simulated by creating 
            multiple `start_producing()`
        """
        publisher = kf.KafkaProducer(bootstrap_servers = params['KAFKA_HOST'])
        step = 0 
        img = cv2.imread(os.path.join('assets', 'FashionMNIST_0.png'), cv2.IMREAD_GRAYSCALE)
        for i, (x, y) in enumerate(test_loader):
            img = x.detach().cpu().numpy()[0][0]
            y = int(y.detach().cpu().numpy()[0])
            payload = {"data" : "Payload data", "image" : img.tolist(), "timestamp": time.time(), "target" : y}
            print("Sending payload")
            publish_payload(params, publisher, payload, params['KAFKA_TOPIC'])
            step += 1
            time.sleep(1)
    
    def start_consuming():
        while True:
            consumer = kf.KafkaConsumer(params['KAFKA_TOPIC'], bootstrap_servers = params['KAFKA_HOST'])
            consume_payload(params, consumer, params['KAFKA_TOPIC'], process_payload)
            time.sleep(1)

    threads = []
    t = threading.Thread(target=start_producing)
    t2 = threading.Thread(target=start_consuming)
    threads.append(t)
    threads.append(t2)
    t.start()
    t2.start()
