#!/usr/bin/env bash
# Explicitly creates the topics this pipeline depends on. The broker runs
# with auto.create.topics.enable=false, so a typo'd topic name fails loudly
# instead of silently creating a new topic — and the topic list (and hence
# the contracts in ARCHITECTURE.md) stays visible in one place.
set -euo pipefail

BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}"
PARTITIONS="${STREAM_PARTITIONS:-3}"
TOPICS_BIN=/opt/kafka/bin/kafka-topics.sh

# name:partitions:retention.ms
# reviews.raw and reviews.predictions MUST have the same partition count:
# both are keyed by review id, so equal counts co-partition them, which is
# what lets consumer_join scale out (see ARCHITECTURE.md "Co-partitioning").
TOPICS=(
  "reviews.raw:${PARTITIONS}:604800000"          # 7 days
  "reviews.predictions:${PARTITIONS}:604800000"  # 7 days
  "reviews.dlq:1:2592000000"                     # 30 days — time to investigate
  "alerts.negative_spike:1:2592000000"           # 30 days
)

for spec in "${TOPICS[@]}"; do
  IFS=: read -r name partitions retention <<<"$spec"
  echo "Ensuring topic: $name (partitions=$partitions, retention.ms=$retention)"
  "$TOPICS_BIN" --bootstrap-server "$BOOTSTRAP" --create --if-not-exists \
    --topic "$name" --partitions "$partitions" --replication-factor 1 \
    --config "retention.ms=$retention"
done

"$TOPICS_BIN" --bootstrap-server "$BOOTSTRAP" --list
echo "Topic setup complete."
