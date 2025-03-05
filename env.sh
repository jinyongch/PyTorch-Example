mkdir -p $HOME/installs
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p $HOME/installs/miniconda3
rm -f Miniconda3-latest-Linux-x86_64.sh
echo ". $HOME/installs/miniconda3/etc/profile.d/conda.sh" >>$HOME/.bashrc
source $HOME/.bashrc

conda create -n mnist python=3.8 -y
conda activate mnist
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
