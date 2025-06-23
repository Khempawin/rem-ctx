#!/bin/bash

python trl_sample.py \
    --base-model pawin205/Qwen-7B-Review-ICLR-GRPO-UR \
    --dataset-id pawin205/iclr-2017-2020-peer-review-with-thinking-trace \
    --data-split long \
    --output-dir saves/REMORX-UR \
    --run-name REMORX-UR \
    --deepspeed-config-path ds_z3_offload_config.json \
    --report-to wandb