import kafka as kf
import json
import time
import os
from constants import params
import threading

def process_payload(message):
    print("Received {}.".format(message))

def publish_payload(params, publisher, payload, topic, **kwargs):
    data = json.dumps(payload).encode("utf-8")
    publisher.send(
        topic, data, **kwargs
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
    while step < 5:
        print("===================================")
        payload = {"data" : "Payload data", "timestamp": time.time()}
        print(f"Sending payload: {payload}.")
        publish_payload(params, publisher, payload, params['KAFKA_TOPIC'])

def start_consuming():
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
