# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import PaiCfg
import torch

import csv
import pandas as pd

import matplotlib.pyplot as plt
import time
import cv2
import threading
import glfw
import matplotlib.animation as animation
class cmd:
    vx = 0.
    vy = 0.
    dyaw = 0.

class mujoco_visual:
    def __init__(self) -> None:
        self.count_lowlevel = 0
        self.close = 1
        self.mujoco_close = 1
        self.stop_event = threading.Event()
        self.vel = [0,0,0]
        self.w = [0,0,0]
    def quaternion_to_euler_array(self,quat):
        # Ensure quaternion is in the correct format [x, y, z, w]
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        
        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)
        
        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        
        # Returns roll, pitch, yaw in a NumPy array in radians
        return np.array([roll_x, pitch_y, yaw_z]) 

    def get_obs(self,data):
        '''Extracts an observation from the mujoco data structure
        '''
        q = data.qpos.astype(np.double)
        dq = data.qvel.astype(np.double)
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
        self.vel = [v[0],v[1],v[2]]
        omega = data.sensor('angular-velocity').data.astype(np.double)
        self.w = omega
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        return (q, dq, quat, v, omega, gvec)

    def pd_control(self,target_q, q, kp, target_dq, dq, kd):
        '''Calculates torques from position commands
        '''
        # print("p:", (target_q - q) * kp )
        # print("d", (target_dq - dq) * kd)
        return (target_q - q) * kp + (target_dq - dq) * kd

    def plot_thread(self):
        plt.ion()  # 打开交互模式
        fig, ax = plt.subplots()
        t, cmd_x,true_x = [], [], []
        line_cmd_x, = ax.plot(t, cmd_x, 'r-')  # 初始化一条红色线条
        line_true_x, = ax.plot(t, true_x, 'b-')  # 初始化一条红色线条

        while not self.stop_event.is_set():
            t.append(self.count_lowlevel*0.001)
            cmd_x.append(cmd.vx)
            true_x.append(self.vel[0])
            # true_x.append(self.vel[1])
            line_cmd_x.set_xdata(t)
            line_cmd_x.set_ydata(cmd_x)
            line_true_x.set_xdata(t)
            line_true_x.set_ydata(true_x)
            ax.relim()  # 重新计算坐标轴范围
            ax.autoscale_view()  # 自动缩放视图
            plt.draw()  # 绘制更新
            plt.pause(0.001)
        plt.savefig('sine_wave.png')  # 保存为 PNG 格式
        plt.ioff()
        plt.close('all')

    def run_mujoco(self,policy, cfg):
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
        model.opt.timestep = cfg.sim_config.dt
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        viewer = mujoco_viewer.MujocoViewer(model, data)
        self.window = viewer.window
        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
        action = np.zeros((cfg.env.num_actions), dtype=np.double)

        hist_obs = deque()
        for _ in range(cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

        for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
            if glfw.window_should_close(self.window):
                print("=============out mujoco==========")
                break
            # Obtain an observation
            q, dq, quat, v, omega, gvec = self.get_obs(data)
            q = q[-cfg.env.num_actions:]
            dq = dq[-cfg.env.num_actions:]
            for i in range(6):
                tmpq = q[i]
                q[i] = q[i+6]
                q[i+6] = tmpq

                tmpdq = dq[i]
                dq[i] = dq[i+6]
                dq[i+6] = tmpdq
            # 1000hz -> 100hz
            if self.count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = self.quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                obs[0, 0] = math.sin(2 * math.pi * self.count_lowlevel * cfg.sim_config.dt  / 0.5)
                obs[0, 1] = math.cos(2 * math.pi * self.count_lowlevel * cfg.sim_config.dt  / 0.5)
                obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
                obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
                obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 29:41] = action
                obs[0, 41:44] = omega
                obs[0, 44:47] = eu_ang

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q = action * cfg.control.action_scale
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            # Generate PD control
            tau = self.pd_control(target_q, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            for i in range(6):
                tmptau = tau[i]
                tau[i] = tau[i+6]
                tau[i+6] = tmptau
            data.ctrl = tau
            
            mujoco.mj_step(model, data)
            viewer.render()
            if self.count_lowlevel>10*1000:
                cmd.vx = 0.3
            if self.count_lowlevel>20*1000:
                cmd.vx = 0.6
            if self.count_lowlevel>40*1000:
                cmd.vx = 0.8
            self.count_lowlevel += 1
        self.stop_event.set()
        viewer.close()

if __name__ == '__main__':
    import argparse
    print(LEGGED_GYM_ROOT_DIR)
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=False,
                        help='Run to load from.',
                        default=f"{LEGGED_GYM_ROOT_DIR}/logs/Pai_ppo/exported/policies/policy_1.pt")
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(PaiCfg):

        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 20

        class robot_config:
            kps = [40,20,20,40,30,10]*(2)
            kds = [1.2,.9,.9,1.2,.9,.6]*(2)
            tau_limit = 40. * np.ones(12, dtype=np.double)

    policy = torch.jit.load(args.load_model)
    a = mujoco_visual()
    matplotlib_thread = threading.Thread(target=a.plot_thread)
    mujoco_thread = threading.Thread(target=a.run_mujoco,args=(policy, Sim2simCfg()))
    matplotlib_thread.start()
    mujoco_thread.start()
    matplotlib_thread.join()
    mujoco_thread.join()
    print("Both threads have finished. Main thread will now terminate.")
