# ATTNChecker

* 姓名: 方泽宇
* 专业: 计算机科学与技术
* 学号: 25303050218
* 实验时间：2026/8/20～2026/9/4

[TOC]

## 环境配置

1. install docker
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker-aliyun.gpg
sudo chmod a+r /etc/apt/keyrings/docker-aliyun.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker-aliyun.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu jammy stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
docker version

sudo usermod -aG docker $USER
newgrp docker     

2. install nvidia-container-toolkit and link docker
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
nvidia-smi

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg2

curl -fsSL https://mirrors.ustc.edu.cn/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://mirrors.ustc.edu.cn/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://nvidia.github.io#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://mirrors.ustc.edu.cn#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemlibctl restart docker

sudo docker pull docker.m.daocloud.io/nvidia/cuda:12.6.0-base-ubuntu24.04 # Daocloud代理拉取实现link检验
sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi

3. pull docker image
sudo docker pull docker.1ms.run/lyh911/attnchk-pytorch:2.0 # 不在Daocloud的白名单上换中间源
sudo docker tag docker.1ms.run/lyh911/attnchk-pytorch:2.0 lyh911/attnchk-pytorch:2.0
docker run --ipc=host --shm-size=512m --gpus all -it --rm lyh911/attnchk-pytorch:2.0

## 核心算法
EEC-ABFT for GEMM

## 完整实验
<!-- mode0 baseline：原始训练/推理，无检测无纠正                           
mode1 naive ABFT：检测纠正能力有，但 full check 多，开销大            
mode2 adaptive ATTNChecker：检测纠正能力有，但通过 pass-checksum 和选择性校验降低开销 -->
1. Computing Overhead
```
sudo docker run --network host --ipc=host --shm-size=512m --gpus all -it --rm -e HF_ENDPOINT=https://hf-mirror.com lyh911/attnchk-pytorch:2.0 # 容器拉取数据集超时, 让容器复用宿主机网络栈
export HF_ENDPOINT=https://hf-mirror.com 
python ./records/cleanRecords.py
python ./ABFT_running_time/gpt2.py
```

2. Detection and Correction Rating 
```
way fault
mkdir -p logs results/recovery
python records/cleanRecords.py
printf '2' > control/AttnChecker_Mod.txt
printf 't' > control/DEBUG.txt
printf 't' > control/Injection.txt
printf '0\n' > control/pos.txt
CUDA_LAUNCH_BLOCKING=1 python runModels/gpt2.py > logs/gpt2_fixed_pos0.log 2>&1
```
没有Injection无法判断Detection rating
运行级检测率
```
mkdir -p results/fixed_pos0                                           
                                                                        
  for i in $(seq 1 100); do                                             
    printf '2' > control/AttnChecker_Mod.txt                            
    printf 't' > control/Injection.txt                                  
    printf 't' > control/DEBUG.txt                                      
    printf '0\n' > control/pos.txt                                      
                                                                        
    python runModels/bertTest.py > results/fixed_pos0/trial_$i.txt 2>&1 
                                                                        
    if grep -Eq '\[(col|row) check\].*(error detected|INF detected|NAN  
    detected|chk inf error detected)' results/fixed_pos0/trial_$i.txt;  
    then                                                                
      echo "$i detected"                                                
    else                                                                
      echo "$i missed"                                                  
    fi                                                                  
  done | tee results/fixed_pos0/summary.txt
```

3. Adaptive ABFT Detection Frequencies
1) mode2:
```
python ./records/cleanRecords.py
python ./ABFT_running_time/gpt2.py > logs/gpt2_mode2_overhead.py
grep -E 'Attention Mechanism Overhead|Training Overhead|ATTNChecker Loss|no ATTNChecker Loss' logs/bert_mode2_overhead.log
```

2) mode1:
```
python ./records/cleanRecords.py
printf '1' > control/AttnChecker_Mod.txt
python ./ABFT_running_time/gpt2.py > logs/gpt2_mode1_overhead.py
grep -E 'Attention Mechanism Overhead|Training Overhead|ATTNChecker Loss|no ATTNChecker Loss' logs/bert_mode1_overhead.log
```

4. Recovery Overhead
1) checkpoint save/load recovery overhead
```
python records/cleanRecords.py
python ./ABFT_running_time/gpt2.py
python records/cleanRecords.py
python ./Checkpoint_time/gpt2.py
```

2) ABFT kernel correction recovery overhead
```
mkdir -p logs results/recovery
python records/cleanRecords.py
printf '2' > control/AttnChecker_Mod.txt
printf 't' > control/DEBUG.txt
printf 't' > control/Injection.txt
printf '0\n' > control/pos.txt
CUDA_LAUNCH_BLOCKING=1 python runModels/gpt2.py > logs/gpt2_abft_recovery.log 2>&1
```

5. Different Batch-size Expriments
```
mkdir -p logs results/batch_size                                                                                      
for bs in 1 2 4 8 16 32; 
> do                                           
> python records/cleanRecords.py                                                                
> sed -E "s/per_device_train_batch_size[[:space:]]*=[[:space:]]*[0-9]+/per_device_train_batch_size = ${bs}/" ABFT_running_time/gpt2.py > /tmp/gpt2_bs_${bs}.py                                                                                 
> python /tmp/gpt2_bs_${bs}.py > logs/gpt2_bs_${bs}.log 2>&1                                                                    
> grep -E "Attention Mechanism Overhead|Training Overhead|ATTNChecker Loss|no ATTNChecker Loss" logs/gpt2_bs_${bs}.log | tee results/batch_size/gpt2_bs_${bs}.txt 
> done
```

6. Custom Optimized Encoder Kernel
<!-- baseline: OptABFT_v3
optimizes: OptABFT_v4
model: gpt2
batch_size: 8 -->
```
mkdir results
python records/cleanRecords.py                                        
printf '2' > control/AttnChecker_Mod.txt                              
printf 't' > control/DEBUG.txt                                        
printf 'f' > control/Injection.txt
python runModels/gpt2.py > results/v4_debug.txt 2>&1
for f in Q K V AS CL OUT preparation cpy BGemmCorrect; 
do 
awk -v name="$f" '{s+=$1;n++} END{if(n) printf "%s avg_ms=%.6f n=%d\n", name, s/n, n; else printf "%s empty\n", name}' results/time/$f.txt                       
done

cp OptABFT_v3/CUDABlas.cu pytorch/aten/src/ATen/cuda/CUDABlas.cu      
cp OptABFT_v3/CUDABlas.h pytorch/aten/src/ATen/cuda/CUDABlas.h        
cp OptABFT_v3/opt_kernels.cu pytorch/aten/src/ATen/cuda/opt_kernels.cu
cp OptABFT_v3/Blas.cpp pytorch/aten/src/ATen/native/cuda/Blas.cpp
cd pytorch
python setup.py develop

python records/cleanRecords.py                                        
printf '2' > control/AttnChecker_Mod.txt                              
printf 't' > control/DEBUG.txt                                        
printf 'f' > control/Injection.txt
python runModels/gpt2.py > results/v3_debug.txt 2>&1
for f in Q K V AS CL OUT preparation cpy BGemmCorrect; 
do 
awk -v name="$f" '{s+=$1;n++} END{if(n) printf "%s avg_ms=%.6f n=%d\n", name, s/n, n; else printf "%s empty\n", name}' results/time/$f.txt                       
done
```

## 结果分析
1. Computing Overhead
per step training:
Training Overhead:  0.12417662657818716                                
ATTNChecker Loss:  0.5328999999999999                                  
no ATTNChecker Loss:  0.5328999999999999
Training Overhead exists but the loss is subtle. 

2. Detection and Correction Rating
Both detection and correction rating are 100% in paper.
But my correction rating is 98.967% and detection rating is 92%.
可能与错误纠正在对角线这一特殊位置，CUDA数值精度等相关

3. Adaptive ABFT Detection Frequencies
mode1_training overhead > mode2_training overhead =>
Adaptive ABFT Detection Frequencies can reduce the training overhead

4. Recovery Overhead
Overhead of Checkpointing:  22.754563590047233
ATTNChecker kernel correction time:  0.5386708320002072 
ATTNChecker kernel is up to 34× overhead reduction compared with CR in paper.
ATTNChecker kernel is up to 42.24205x overhead reduction compared with CR in my experiment.
CR时间明显偏大,可能与cleanRecords有关(好像漏了一次记录清空)冗余数据加载时间使得时间偏大

5. Different Batch-size Expriments
batch_size = 16                             
Attention Mechanism Overhead:  -0.08521124187890557  
测量噪声, GPU频率/温度状态, CUDA kernel缓存等...

6. Custom Optimized Encoder Kernel                         
v4_avg < v3_avg =>
custom kernel can reduce checksum encoding and ABFT overhead

## 论文不足和改进
不同部位检测中为了降低检测成本在所有section延迟检测,
使得deterministic -> nondeterministic, 反而增加了检测成本
考虑是否可以权衡或分类检测
