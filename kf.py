import kafka as kf
import json
import time
import os
from constants import params
import threading
import cv2
import io
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

def publish_payload(params, publisher, payload, topic, **kwargs):
    payload = json.dumps(payload).encode("utf-8")
    publisher.send(
        topic, payload, **kwargs
    )
    publisher.flush()

def consume_payload(params, consumer, topic, callback, **kwargs):
    # Method has different arguments than that for pubsub
    # because of difference in Kafka API structure
    for msg in consumer:
        callback(msg)

"""
    Sample code for utilisation of methods mentioned above
"""

def start_producing():
    publisher = kf.KafkaProducer(bootstrap_servers = params['KAFKA_HOST'])
    step = 0
    img = cv2.imread(os.path.join('data', 'test', 'FashionMNIST_0.png'), cv2.IMREAD_GRAYSCALE)
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
