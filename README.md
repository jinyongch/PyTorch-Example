# Pytorch Example

## Environment

```bash
git clone git@github.com:jinyongch/PyTorch-Example.git

conda create -n mnist python=3.8 -y
conda activate mnist
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## MNIST

```bash
CUDA_VISIBLE_DEVICES=1 python main.py
```

## Distributed Data Parallel

1. Setting up the Distributed Environment for Distributed Training

    ```python
    dist.init_process_group(backend="nccl", init_method="env://")
    main()
    dist.destroy_process_group()
    ```

2. Global `rank` and `world_size` for Dataset Partitioning

    ```python
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"]) 

    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler)
    ```

3. `local_rank` for Model Parallelization and Gradient Synchronization Across GPUs

    ```python
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    model = Net().to(device)
    model = DDP(model, device_ids=[local_rank])

    data, target = data.to(device), target.to(device)
    ```

4. Distributed Training

    ```bash
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
    ```
