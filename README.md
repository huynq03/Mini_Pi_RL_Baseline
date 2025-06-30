![sim2sim](https://github.com/HighTorque-Robotics/livelybot_pi_rl_baseline/blob/release_v1.0.0/tinywow_%E9%A3%9E%E4%B9%A620250630-092216_82033498.gif)

# Pi_rl_baseline

The baseline provided a reinforcement learning environment based on NVIDIA Isaac Gym. For the Pi Humanoid Robots from the HighTorque Robotics , Pi_rl_baseline also includes sim2sim framework from Isaac Gym to Mujoco, enabling users to validate their trained policies in different simulations to ensure policy robustness and generalization capabilities. 

## Installation

1. Use `miniconda` or `anaconda` to create a virtual environment `conda create -n pi_env python=3.8`.
2. Use `apt` to install nvidia driver `sudo apt install nvidia-driver-525`, the driver version has to be at least 515. Installing higher version is also viable, as the driver is backward compatible. After installation, check the graphic driver's CUDA version using `nvidia-smi` . As shown in the picture, the CUDA version is 12.4, driver version is 550. 

   ![1730344376083](image/README/1730344376083.png)
3. Install the latest version of `Pytorch` : Visit `Pytorch` website https://pytorch.org/. For `Package `choose `Conda `, for `Compute Platform` choose suitable `CUDA` version. `CUDA` is backward compatible, but not forward compatible software library, so the chosen `CUDA` version needs to be smaller than the computer's installed version. 

   ![1730344921405](image/README/1730344921405.png)

   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   ```
4. Use `conda` to install numpy `conda install numpy=1.23`.
5. Install `Isaac Gym`:

   - Visit Nvidia official website, download and install `Isaac Gym Preview 4` `https://developer.nvidia.com/isaac-gym`.
   - Activate conda environment, then access `isaacgym` repository to install `cd isaacgym/python && pip install -e .`
   - Run the example python script to test whether the environment is installed successfully: `cd examples && python 1080_balls_of_solitude.py`.
   - Please read `isaacgym/docs/index.html` to troubleshoot. 
6. Install this baseline:

   - Clone this repository `git clone https://github.com/HighTorque-Locomotion/pi_rl_baseline.git`.
   - `cd pi_rl_baseline && pip install -e .`

## Usage Guide

#### Examples

```bash
# Use 4096 environments, and using "v1" as training version to do PPO policy training. 
# This command will initiate the robot's training task. 
python scripts/train.py --task=pai_ppo --run_name v1 --headless --num_envs 4096

# Evaluate the trained policy
# This command will load "v1" policy for performance evaluation under its environment. 
# In addition, it will automatically exports a JIT model suitable for deployment purposes.
python scripts/play.py --task=pai_ppo --run_name v1

# Use Mujoco to achieve sim2sim
python scripts/sim2sim.py --load_model /path/to/logs/Pai_ppo/exported/policies/policy_torch.pt

# Run the trained policy provided by us
python scripts/sim2sim.py --load_model /path/to/logs/Pai_ppo/exported/policies/policy_example.pt
```

#### Parameters

- **CPU and GPU Usage**: Use CPU to run the simulation, set `--sim_device=cpu` and `--rl_device=cpu` simultaneously. Use specific GPU to run the simulation, set `--sim_device=cuda:{0,1,2...}` and `--rl_device={0,1,2...}` simultaneously. Please note: `CUDA_VISIBLE_DEVICES` is not applicable, and the setting that match `--sim_device` and `--rl_device` is very important. 
- **Headless Operation**: use `--headless` parameters to run without rendering.
- **Rendering Control**: during training, press 'v' to open or close rendering. 
- **Policy Location**: Trained model save to `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`.


#### Command-Line Arguments

Run RL training, please refer to `humanoid/utils/helpers.py`.
Run sim2sim, please refer to `humanoid/scripts/sim2sim.py`.


1. Every environment relies on an `env` file (`legged_robot.py`) and a `config` file (`legged_robot_config.py`). The latter includes two classes: `LeggedRobotCfg` (which includes all environment parameters) and `LeggedRobotCfgPPO` (which includes all training parameters). 
2. Both `env` and `config` classes use inheritance. 
3. In `cfg`, a non-zero reward specified in contributes the correspondingly named function to the total reward. 
4. Must use `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)` to register task. Registration can happen within `envs/__init__.py` or outside the repository. 

## Add a new environment

Basic environment "legged_robot" creates a construct a rough terrain locomotion task. The corresponding configuration does not specify the robot assets (URDF/MJCF) and reward scale. 

1. If you need to add a new environment, please create a new repository under "envs/", and includes configuration file "<your_env>_config.py" within. New configurations should inherit from existing environment configurations. 
2. If you propose to use a new robot: 

   - Insert corresponding assets into the "resources/" repository. 
   - In "cfg" files, set the path to the asset, define the body name, default_joint_positions and PD gains. Specify the desired 'train_cfg' and the name of the environment (python class).
   - In "train_cfg" , set "experiment_name" and "run_name". 

3. If needed, please create your environment in "<your_env>.py". Inherit from an existing environment, overriding required functionality and/or adding your bonus functionality. 
4. Register your environment in `humanoid/envs/__init__.py` . 
5. Modify or adjust other parameter in `cfg` or `cfg_train` according to your need. To remove the reward, set its scale to zero. Avoid modifying other environments' parameters!
6. If you want your new robots/environments to implement sim2sim, you might need to change `humanoid/scripts/sim2sim.py`: 

   - Check the robot joint mapping between MJCF and URDF. 
   - Change the initial joint positions of the robot according to your trained policy. 

## Acknowledgment

The accomplishment of pi_rl_baseline relied on the resource from the [legged_gym](https://github.com/leggedrobotics/legged_gym) projec


# Pi_rl_baseline（Chinese translation）
该基线工作提供了一个基于 NVIDIA Isaac Gym 的强化学习环境，对 高擎机电的双足机器人 Pi Pi_rl_baseline 还整合了从 Isaac Gym 到 Mujoco 的sim2sim框架，使用户能够在不同的物理模拟中验证训练得到的策略，以确保策略的稳健性和泛化能力。

## 安装

1. 使用 `miniconda` 或 `anaconda` 创建一个虚拟环境 `conda create -n pi_env python=3.8`.
2. 使用 `apt` 安装nvidia显卡驱动 `sudo apt install nvidia-driver-525`,驱动版本至少为515，因为驱动是向下兼容的，所以也可以安装更高版本的驱动。安装完成后，在命令行中使用命令 `nvidia-smi` 以查看驱动的CUDA版本。可以看到示例图片中的CUDA版本为12.4，驱动版本为550。

   ![1730344376083](image/README/1730344376083.png)
3. 安装最新版本的 `Pytorch` : 进入 `Pytorch` 官网 https://pytorch.org/ ，`Package `选项选择 `Conda `,`Compute Platform`选择合适的 `CUDA` 版本。`CUDA` 是一个向下兼容，但不向上兼容的软件库，所以所选择的 `CUDA` 版本要小于等于电脑安装的版本。

   ![1730344921405](image/README/1730344921405.png)

   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   ```
4. 使用 `conda` 安装numpy `conda install numpy=1.23`.
5. 安装 `Isaac Gym`:

   - 在Nvidia官网下载并安装 `Isaac Gym Preview 4` `https://developer.nvidia.com/isaac-gym`.
   - 激活conda环境，并进入 `isaacgym`的包中进行安装 ： `cd isaacgym/python && pip install -e .`
   - 可以通过运行自带的示例脚本，测试环境安装是否成功： `cd examples && python 1080_balls_of_solitude.py`.
   - 请参阅 `isaacgym/docs/index.html` 以进行故障排除。
6. 安装本baseline:

   - 克隆此仓库： `git clone https://github.com/HighTorque-Locomotion/pi_rl_baseline.git`.
   - `cd pi_rl_baseline && pip install -e .`

## Usage Guide

#### Examples

```bash
# 使用 4096 个环境，并以“v1”为训练版本进行 PPO policy 训练
# 该命令将会开始机器人的训练任务.
python scripts/train.py --task=pai_ppo --run_name v1 --headless --num_envs 4096

# 评估训练好的policy
# 此命令将会加载“v1”policy以在其环境中进行性能评估。
# 此外，它还会自动导出适合部署目的的 JIT 模型。
python scripts/play.py --task=pai_ppo --run_name v1

# 通过使用Mujoco实现sim2sim
python scripts/sim2sim.py --load_model /path/to/logs/Pai_ppo/exported/policies/policy_torch.pt

# 运行我们提供的训练好的policy
python scripts/sim2sim.py --load_model /path/to/logs/Pai_ppo/exported/policies/policy_example.pt
```

#### Parameters

- **CPU and GPU Usage**: 使用CPU运行仿真, 同时设置 `--sim_device=cpu` 和 `--rl_device=cpu`. 使用指定GPU运行仿真，同时设置 `--sim_device=cuda:{0,1,2...}` 和 `--rl_device={0,1,2...}`. 请注意，`CUDA_VISIBLE_DEVICES` 不适用，并且匹配 `--sim_device` 和 `--rl_device` 的设置至关重要。
- **Headless Operation**: 使用 `--headless` 参数用于无渲染运行.
- **Rendering Control**: 在训练期间按 `v` 键开启或关闭渲染.
- **Policy Location**: 训练好的模型保存在 `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`.


#### Command-Line Arguments

进行RL训练，请参考 `humanoid/utils/helpers.py`.
进行sim2sim，请参考 `humanoid/scripts/sim2sim.py`.


1. 每个环境都依赖于一个 `env` 文件（`legged_robot.py`）和一个 `config` 文件（`legged_robot_config.py`）。后者包含两个类：`LeggedRobotCfg`（包含所有环境参数）和 `LeggedRobotCfgPPO`（表示所有训练参数）。
2. `env` 和 `config` 类都使用继承。
3. `cfg` 中指定的非零奖励将相应名称的函数贡献给总奖励。
4. 必须使用 `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)` 注册任务。注册可能发生在 `envs/__init__.py` 内，也可能发生在此存储库之外。

## Add a new environment

基础环境“legged_robot”构建了一个崎岖地形运动任务。相应的配置未指定机器人资产（URDF/MJCF）和奖励量表。

1. 如果您需要添加新环境，请在“envs/”目录中创建一个新文件夹，其中包含名为“<your_env>_config.py”的配置文件。新配置应继承自现有环境配置。
2. 如果提议使用新机器人：

   - 将相应的资产插入“resources/”文件夹中。
   - 在“cfg”文件中，设置资产的路径，定义主体名称、default_joint_positions 和 PD 增益。指定所需的“train_cfg”和环境的名称（python 类）。
   - 在“train_cfg”中，设置“experiment_name”和“run_name”。

3. 如果需要，请在“<your_env>.py”中创建您的环境。从现有环境继承，覆盖所需功能和/或添加您的奖励功能。
4. 在 `humanoid/envs/__init__.py` 中注册您的环境。
5. 根据需求修改或调整 `cfg` 或 `cfg_train` 中的其他参数。要删除奖励，请将其比例设置为零。避免修改其他环境的参数！
6. 如果您想要新的机器人/环境来执行 sim2sim，您可能需要修改 `humanoid/scripts/sim2sim.py`：

   - 检查 MJCF 和 URDF 之间的机器人关节映射。
   - 根据您训练的策略更改机器人的初始关节位置。

## Acknowledgment

pai_rl_baseline 的实现依赖于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 项目的资源。
