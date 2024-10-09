
## **High-Torque Pi RL Baseline**

## **基础环境配置** ：

1. 安装conda：https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
2. 创建conda环境：
   推荐使用python3.8，myenv为环境名称，可自定义。

```Plain
 conda create -n myenv python=3.8
```

3. 在命令行中输出指令，查看当前主机的cuda版本：

```Plain
nvidia-smi
```

如图CUDA版本为12.5，如果输入该指令没有成功显示出来如图的参数，请给自己电脑安装CUDA驱动。

![](https://lingdongfangcheng.feishu.cn/space/api/box/stream/download/asynccode/?code=MzcyNjBmOWVjNTU3NTA0ZWUxNWZlMjQwMTI3MzcwZTNfS1JlbkpTcmVxZDYzbTNWNDNYb1Y1b1lXRUk3NjZRZ1ZfVG9rZW46SjNBSmJnUGNyb1dQQ3l4VXI1U2NJYWswblhmXzE3Mjg0NDMxNDU6MTcyODQ0Njc0NV9WNA)

CUDA驱动安装可以在命令行中使用apt安装，推荐使用最新版本

```Plain
sudo apt install nvidia-driver-560
```

![](https://lingdongfangcheng.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDdmZWI5OTM2OTI0M2EzMDU4OThjYjE1NDUxMzU4NTVfd2d4Z29ubldYdlNnemxrb0F2b2JYWTBISzRSNmtXQUNfVG9rZW46VnFZSmJjSGRpb2kzOTd4amxPbWNUcDFtbjJlXzE3Mjg0NDMxNDU6MTcyODQ0Njc0NV9WNA)

4. 在conda环境中安装系列包：
   首先打开一个终端，在命令行中输入指令以激活conda环境：conda activate myenv

   1. pytorch：
      打开pytorch官网：https://pytorch.org/

      根据电脑的CUDA版本进行安装，复制下面的指令到命令行中进行安装。CUDA具有向下兼容的特性，在选择cumpute platform时，所选的CUDA版本不得高于电脑的CUDA版本。

      ![](https://lingdongfangcheng.feishu.cn/space/api/box/stream/download/asynccode/?code=YjA0NzA2ODA2ZDQ3ZTY4ZjQ2NWU1NDMxNjIwZjVlOGVfdDl4R2JwSlZYb2hRNlJHSXhwcjVUZnlSUjdMRjFLRjRfVG9rZW46R0kzWmJxY2xlb1FkM2l4anpUVmNpdlZUbmtoXzE3Mjg0NDMxNDU6MTcyODQ0Njc0NV9WNA)
   2. Isaac Gym Preview 4:
      进入官网下载：https://developer.nvidia.com/isaac-gym
      将会得到一个压缩文件为 `IsaacGym_Preview_4_Package.tar.gz`然后打开压缩包解压，得到文件夹 `isaacgym`，进入解压后的文件夹，激活conda环境后进行安装。

   在命令行中输入指令：

   ```Plain
   cd isaacgym/python
   conda activate myenv
   pip install -e .
   ```

   1. pi_rl_baseline:
      Git 源代码,并使用pip进行安装：

   ```Bash
   git clone https://github.com/HighTorque-Locomotion/pi_rl_baseline.git
   cd pi_rl_baseline 
   pip install -e .
   ```

## **使用指南：**

  示例：

```Bash
# 启动 4096 个环境中的 PPO 策略训练 
python scripts/train.py --headless --num_envs 4096
# 评估训练好的最新的策略，并自动导出一个适合部署的JIT模型
python scripts/play.py
# sim2sim 使用导出的JIT模型，在mujoco中进行 sim-to-sim测试评估
python scripts/sim2sim.py --load_model /path/to/logs/Pai_ppo/exported/policies/policy_1.pt
```
