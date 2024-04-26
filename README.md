# Basic MNIST Example

```bash
git clone git@github.com:jinyongch/MNIST.git

conda create -n mnist python=3.8 -y
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt

python main.py
# CUDA_VISIBLE_DEVICES=1 python main.py
```
