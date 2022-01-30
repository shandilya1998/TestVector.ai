import os
import json
from constants import params
from google.cloud import pubsub_v1 as ps
from concurrent.futures import TimeoutError
import time

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = params['GOOGLE_APPLICATION_CREDENTIALS']

def process_payload(message):
    print(f"Received {message.data}.")
    message.ack()    

# producer function to push a message to a topic
def publish_payload(params, publisher, payload, topic, **kwargs):        
    topic_path = publisher.topic_path(params['PUB_SUB_PROJECT'], topic)        
    data = json.dumps(payload).encode("utf-8")           
    future = publisher.publish(topic_path, data=data, **kwargs)
    print("Pushed message to topic.")   

# consumer function to consume messages from a topics for a given timeout period
def consume_payload(params, subscription, callback, subscriber, **kwargs):
    subscription_path = subscriber.subscription_path(params['PUB_SUB_PROJECT'], subscription)
    print(f"Listening for messages on {subscription_path}..\n")
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    # Wrap subscriber in a 'with' block to automatically call close() when done.
    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.                
            streaming_pull_future.result(timeout = params['TIMEOUT'])
        except TimeoutError:
            streaming_pull_future.cancel()


if __name__ == '__main__':
    publisher = ps.PublisherClient()
    step = 0
    while(step < 5):
        print("===================================")
        payload = {"data" : "Payload data", "timestamp": time.time()}
        print(f"Sending payload: {payload}.")
        publish_payload(params, publisher, payload, params['PUB_SUB_TOPIC'])
        subscriber = ps.SubscriberClient()
        consume_payload(params, params['PUB_SUB_SUBSCRIPTION'], process_payload, subscriber)
        step += 1