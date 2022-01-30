#!/bin/bash
kafka_2.13-3.1.0/bin/zookeeper-server-start.sh \
    -daemon \
    kafka_2.13-3.1.0/config/zookeeper.properties 
kafka_2.13-3.1.0/bin/kafka-server-start.sh \
    -daemon \
    kafka_2.13-3.1.0/config/server.properties

