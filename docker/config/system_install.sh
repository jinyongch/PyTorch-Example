cp /etc/apt/sources.list /etc/apt/sources.list.bak
cat <<EOF | tee /etc/apt/sources.list
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
EOF

apt update && apt install -y openssh-server zsh vim tmux autojump sudo tree bat git gpustat net-tools iputils-ping telnet iproute2 wget

echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main" | tee -a /etc/apt/sources.list
apt update && apt install -y libc6
strings /lib/x86_64-linux-gnu/libc.so.6 | grep GLIBC_

cat <<EOF | tee -a /etc/environment
DOCKER_CONTAINER=1
EOF

cat <<EOF | tee -a /etc/ssh/sshd_config
Port ${HOST_UID}
AuthenticationMethods publickey
EOF

service ssh restart

groupadd --gid ${HOST_UID} ${HOST_USER}
useradd --uid ${HOST_UID} --gid ${HOST_UID} ${HOST_USER}
echo "${HOST_USER}:${HOST_USER}" | chpasswd

groupadd --gid 8888 user
usermod -aG user ${HOST_USER}

usermod -aG sudo ${HOST_USER}
echo "${HOST_USER} ALL=(ALL) NOPASSWD: ALL" >>/etc/sudoers

chsh -s $(which zsh) ${HOST_USER}
