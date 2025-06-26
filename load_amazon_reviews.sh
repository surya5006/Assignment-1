#!/usr/bin/env bash
# ===============================================================
#  Load Amazon Reviews 2023 – Subscription_Boxes into HDFS

set -e          # Exit on any error
set -u          # Treat unset variables as an error

echo "▶ Starting Hadoop daemons (HDFS & YARN) ..."
start-dfs.sh          # launches NameNode, DataNode(s)
start-yarn.sh         # launches ResourceManager, NodeManager(s)

echo "▶ Creating target directory in HDFS ..."
hdfs dfs -mkdir -p /amazon_reviews

echo "▶ Uploading dataset files to HDFS ..."
hdfs dfs -put -f Subscription_Boxes.jsonl /amazon_reviews/
hdfs dfs -put -f meta_Subscription_Boxes.jsonl /amazon_reviews/

echo "▶ Verifying upload ..."
hdfs dfs -ls -h /amazon_reviews
