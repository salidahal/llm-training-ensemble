#!/bin/bash
# Run training experiments across different data budgets
# 50M (1 shard), 100M (2 shards), 200M (4 shards)
# 4 rounds each, 3 epochs per run

set -e

EPOCHS=3
BATCH_SIZE=128
GLOBAL_BATCH_SIZE=512
VAL_EVERY=200

# Data budget configs: name, num_shards
declare -a BUDGETS=(
    "50m:1"
    "100m:2"
    "200m:4"
)

for budget in "${BUDGETS[@]}"; do
    IFS=':' read -r name shards <<< "$budget"

    for i in $(seq 1 4); do
        RUN_NAME="model_${name}_${i}"
        echo "============================================================"
        echo "Starting: ${RUN_NAME} (${name} tokens, shards=${shards}, round ${i}/4)"
        echo "============================================================"

        python src/train.py \
            --epochs $EPOCHS \
            --shards $shards \
            --batch-size $BATCH_SIZE \
            --global-batch-size $GLOBAL_BATCH_SIZE \
            --val-every $VAL_EVERY \
            --no-test \
            --sample-every 0 \
            --run-name "$RUN_NAME"

        echo "Finished: ${RUN_NAME}"
        echo ""
    done
done

echo "All experiments complete!"
