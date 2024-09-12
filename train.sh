#!/usr/bin/bash

if [[ $(hostname) == "husky" ]]; then
    NETWORK="br0"
elif [[ $(hostname) == "panda" ]]; then
    NETWORK="enp7s0"
else
    NETWORK=$(ip -o link show | awk -F': ' '{ print $2 }' | grep -vE "lo|br|docker|veth")
fi

TORCHRUN="${HOME}/installs/miniconda3/envs/mnist/bin/torchrun"

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_SOCKET_IFNAME=${NETWORK} \
    $TORCHRUN \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="10.6.176.116" \
    --master_port=12345 \
    train.py
