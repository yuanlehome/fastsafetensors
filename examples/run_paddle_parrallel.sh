# !/usr/bin/env python3
PIDS=()

runner="python -m paddle.distributed.launch"
# runner="torchrun"

cd paddle_case
${runner} --nnodes=2 --master=127.0.0.1:12345 --rank=0 run_parallel.py &
PIDS+=($!)
${runner} --nnodes=2 --master=127.0.0.1:12345 --rank=1 run_parallel.py &
PIDS+=($!)
wait "${PIDS[@]}"