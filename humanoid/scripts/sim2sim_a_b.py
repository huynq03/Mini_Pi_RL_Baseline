import math
from pathlib import Path
import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import glfw
import time


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0


class mujoco_visual:
    def __init__(
        self,
        target_x,
        target_y,
        max_vx=0.5,
        max_wz=1.0,
        arrival_distance=0.15,
    ):
        self.count_lowlevel = 0

        # Target B in MuJoCo WORLD coordinates
        self.target_x = target_x
        self.target_y = target_y

        # Navigation parameters
        self.max_vx = max_vx
        self.max_wz = max_wz
        self.arrival_distance = arrival_distance

        # Controller gains
        self.kp_distance = 0.8
        self.kp_yaw = 2.0

        self.reached_target = False

    @staticmethod
    def wrap_to_pi(angle):
        """
        Convert angle to [-pi, pi].
        """
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def quaternion_to_euler_array(self, quat):
        """
        quat format:
            [x, y, z, w]

        return:
            [roll, pitch, yaw]
        """
        x, y, z, w = quat

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z])

    def get_obs(self, data):
        q = data.qpos.astype(np.double)
        dq = data.qvel.astype(np.double)

        # MuJoCo sensor probably gives:
        # [w, x, y, z]
        #
        # scipy requires:
        # [x, y, z, w]
        quat = (
            data.sensor("orientation")
            .data[[1, 2, 3, 0]]
            .astype(np.double)
        )

        r = R.from_quat(quat)

        # World velocity -> robot/body velocity
        v = r.apply(
            data.qvel[:3],
            inverse=True,
        ).astype(np.double)

        omega = (
            data.sensor("angular-velocity")
            .data
            .astype(np.double)
        )

        gvec = r.apply(
            np.array([0.0, 0.0, -1.0]),
            inverse=True,
        ).astype(np.double)

        return q, dq, quat, v, omega, gvec

    def pd_control(
        self,
        target_q,
        q,
        kp,
        target_dq,
        dq,
        kd,
    ):
        return (
            (target_q - q) * kp
            + (target_dq - dq) * kd
        )

    def navigate_to_target(self, data, quat):
        """
        High-level A -> B navigation.

        A = current robot world position
        B = self.target_x, self.target_y

        Generates:
            cmd.vx
            cmd.vy
            cmd.dyaw
        """

        # --------------------------------------------------
        # 1. Current robot position in WORLD frame
        # --------------------------------------------------

        current_x = data.qpos[0]
        current_y = data.qpos[1]

        # --------------------------------------------------
        # 2. Vector from current position to target
        # --------------------------------------------------

        error_x = self.target_x - current_x
        error_y = self.target_y - current_y

        distance = math.sqrt(
            error_x**2 + error_y**2
        )

        # --------------------------------------------------
        # 3. Current robot yaw
        # --------------------------------------------------

        euler = self.quaternion_to_euler_array(quat)
        current_yaw = euler[2]

        # --------------------------------------------------
        # 4. Desired heading toward B
        #
        # atan2:
        #
        #       target
        #          *
        #         /
        #        /
        # robot *
        #
        # --------------------------------------------------

        desired_yaw = math.atan2(
            error_y,
            error_x,
        )

        yaw_error = self.wrap_to_pi(
            desired_yaw - current_yaw
        )

        # --------------------------------------------------
        # 5. Arrived at B
        # --------------------------------------------------

        if distance < self.arrival_distance:

            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.dyaw = 0.0

            if not self.reached_target:
                print()
                print("==============================")
                print("TARGET REACHED")
                print(
                    f"Position: "
                    f"x={current_x:.3f}, "
                    f"y={current_y:.3f}"
                )
                print(
                    f"Target:   "
                    f"x={self.target_x:.3f}, "
                    f"y={self.target_y:.3f}"
                )
                print(
                    f"Error: {distance:.3f} m"
                )
                print("==============================")

                self.reached_target = True

            return

        self.reached_target = False

        # --------------------------------------------------
        # 6. Yaw controller
        # --------------------------------------------------

        yaw_cmd = self.kp_yaw * yaw_error

        yaw_cmd = np.clip(
            yaw_cmd,
            -self.max_wz,
            self.max_wz,
        )

        cmd.dyaw = float(yaw_cmd)

        # --------------------------------------------------
        # 7. Forward velocity controller
        # --------------------------------------------------

        speed = self.kp_distance * distance

        speed = np.clip(
            speed,
            0.0,
            self.max_vx,
        )

        # If robot is facing very far away from target,
        # rotate first instead of walking.
        if abs(yaw_error) > math.radians(45):
            speed = 0.0

        else:
            # Reduce forward speed when heading error exists
            speed *= max(
                0.0,
                math.cos(yaw_error),
            )

        cmd.vx = float(speed)

        # We don't need lateral walking for simple A -> B
        cmd.vy = 0.0

        # --------------------------------------------------
        # Debug output
        # --------------------------------------------------

        if self.count_lowlevel % 500 == 0:
            print(
                f"A=({current_x:+.2f}, "
                f"{current_y:+.2f}) "
                f"B=({self.target_x:+.2f}, "
                f"{self.target_y:+.2f}) "
                f"dist={distance:.2f} "
                f"yaw={math.degrees(current_yaw):+.1f}° "
                f"target_yaw={math.degrees(desired_yaw):+.1f}° "
                f"err={math.degrees(yaw_error):+.1f}° "
                f"vx={cmd.vx:.2f} "
                f"wz={cmd.dyaw:.2f}"
            )

    def run_mujoco(self, policy, cfg):

        # --------------------------------------------------
        # Load MuJoCo robot
        # --------------------------------------------------

        model = mujoco.MjModel.from_xml_path(
            cfg.sim_config.mujoco_model_path
        )

        model.opt.timestep = cfg.sim_config.dt

        data = mujoco.MjData(model)

        mujoco.mj_step(model, data)

        viewer = mujoco_viewer.MujocoViewer(
            model,
            data,
        )

        # --------------------------------------------------
        # Policy variables
        # --------------------------------------------------

        target_q = np.zeros(
            cfg.env.num_actions,
            dtype=np.double,
        )

        action = np.zeros(
            cfg.env.num_actions,
            dtype=np.double,
        )

        hist_obs = deque()

        for _ in range(cfg.env.frame_stack):
            hist_obs.append(
                np.zeros(
                    [1, cfg.env.num_single_obs],
                    dtype=np.double,
                )
            )

        print()
        print("==============================")
        print("A -> B navigation")
        print(
            f"Start A = "
            f"({data.qpos[0]:.3f}, "
            f"{data.qpos[1]:.3f})"
        )
        print(
            f"Target B = "
            f"({self.target_x:.3f}, "
            f"{self.target_y:.3f})"
        )
        print("==============================")
        print()

        total_steps = int(
            cfg.sim_config.sim_duration
            / cfg.sim_config.dt
        )

        for _ in tqdm(
            range(total_steps),
            desc="Simulating...",
        ):

            if glfw.window_should_close(
                viewer.window
            ):
                break

            # --------------------------------------------------
            # Read robot state
            # --------------------------------------------------

            (
                q,
                dq,
                quat,
                v,
                omega,
                gvec,
            ) = self.get_obs(data)

            # --------------------------------------------------
            # A -> B controller
            #
            # This replaces keyboard commands.
            # --------------------------------------------------

            self.navigate_to_target(
                data,
                quat,
            )

            # --------------------------------------------------
            # Select only 12 joint states
            # --------------------------------------------------

            q = q[-cfg.env.num_actions:]
            dq = dq[-cfg.env.num_actions:]

            # MuJoCo joint order -> policy order
            for i in range(6):
                q[i], q[i + 6] = (
                    q[i + 6],
                    q[i],
                )

                dq[i], dq[i + 6] = (
                    dq[i + 6],
                    dq[i],
                )

            # --------------------------------------------------
            # Run RL policy at 50 Hz
            # --------------------------------------------------

            if (
                self.count_lowlevel
                % cfg.sim_config.decimation
                == 0
            ):

                obs = np.zeros(
                    [1, cfg.env.num_single_obs],
                    dtype=np.float32,
                )

                eu_ang = (
                    self.quaternion_to_euler_array(
                        quat
                    )
                )

                eu_ang[
                    eu_ang > math.pi
                ] -= 2 * math.pi

                # gait phase
                obs[0, 0] = math.sin(
                    2
                    * math.pi
                    * self.count_lowlevel
                    * cfg.sim_config.dt
                    / 0.5
                )

                obs[0, 1] = math.cos(
                    2
                    * math.pi
                    * self.count_lowlevel
                    * cfg.sim_config.dt
                    / 0.5
                )

                # desired velocity commands
                obs[0, 2] = (
                    cmd.vx
                    * cfg.normalization.obs_scales.lin_vel
                )

                obs[0, 3] = (
                    cmd.vy
                    * cfg.normalization.obs_scales.lin_vel
                )

                obs[0, 4] = (
                    cmd.dyaw
                    * cfg.normalization.obs_scales.ang_vel
                )

                # joint position
                obs[0, 5:17] = (
                    q
                    * cfg.normalization.obs_scales.dof_pos
                )

                # joint velocity
                obs[0, 17:29] = (
                    dq
                    * cfg.normalization.obs_scales.dof_vel
                )

                # previous actions
                obs[0, 29:41] = action

                # IMU angular velocity
                obs[0, 41:44] = omega

                # orientation
                obs[0, 44:47] = eu_ang

                obs = np.clip(
                    obs,
                    -cfg.normalization.clip_observations,
                    cfg.normalization.clip_observations,
                )

                # --------------------------------------------------
                # Frame stacking
                # --------------------------------------------------

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros(
                    [1, cfg.env.num_observations],
                    dtype=np.float32,
                )

                for i in range(
                    cfg.env.frame_stack
                ):
                    start = (
                        i
                        * cfg.env.num_single_obs
                    )

                    end = (
                        (i + 1)
                        * cfg.env.num_single_obs
                    )

                    policy_input[
                        0,
                        start:end,
                    ] = hist_obs[i][0, :]

                # --------------------------------------------------
                # Neural network
                # --------------------------------------------------

                with torch.no_grad():
                    action[:] = (
                        policy(
                            torch.tensor(
                                policy_input
                            )
                        )[0]
                        .cpu()
                        .numpy()
                    )

                action[:] = np.clip(
                    action,
                    -cfg.normalization.clip_actions,
                    cfg.normalization.clip_actions,
                )

                target_q = (
                    action
                    * cfg.control.action_scale
                )

            # --------------------------------------------------
            # PD controller
            # --------------------------------------------------

            target_dq = np.zeros(
                cfg.env.num_actions,
                dtype=np.double,
            )

            tau = self.pd_control(
                target_q,
                q,
                cfg.robot_config.kps,
                target_dq,
                dq,
                cfg.robot_config.kds,
            )

            tau = np.clip(
                tau,
                -cfg.robot_config.tau_limit,
                cfg.robot_config.tau_limit,
            )

            # Policy joint order -> MuJoCo joint order
            for i in range(6):
                tau[i], tau[i + 6] = (
                    tau[i + 6],
                    tau[i],
                )

            data.ctrl[:] = tau

            # --------------------------------------------------
            # Physics simulation
            # --------------------------------------------------

            mujoco.mj_step(
                model,
                data,
            )

            # Optional real-time-ish slowdown
            time.sleep(0.001)

            if (
                self.count_lowlevel
                % cfg.sim_config.decimation
                == 0
            ):
                viewer.render()

            self.count_lowlevel += 1

        viewer.close()


if __name__ == "__main__":

    import argparse

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    parser = argparse.ArgumentParser(
        description="Mini Pi A -> B navigation"
    )

    parser.add_argument(
        "--load_model",
        type=str,
        default=str(
            project_root
            / "logs/Pai_ppo/exported/policies/policy_1.pt"
        ),
    )

    # ----------------------------------------------
    # Target B
    # ----------------------------------------------

    parser.add_argument(
        "--target_x",
        type=float,
        default=2.0,
        help="Target world X coordinate",
    )

    parser.add_argument(
        "--target_y",
        type=float,
        default=0.0,
        help="Target world Y coordinate",
    )

    parser.add_argument(
        "--max_vx",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max_wz",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    class Sim2simCfg:

        class env:
            frame_stack = 15
            num_single_obs = 47

            num_observations = (
                frame_stack
                * num_single_obs
            )

            num_actions = 12

        class normalization:

            class obs_scales:
                lin_vel = 2.0
                ang_vel = 1.0
                dof_pos = 1.0
                dof_vel = 0.05

            clip_observations = 18.0
            clip_actions = 18.0

        class control:
            action_scale = 0.25

        class sim_config:

            mujoco_model_path = str(
                project_root
                / "resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1.xml"
            )

            sim_duration = 60.0

            dt = 0.001
            decimation = 20

        class robot_config:

            kps = np.array(
                [40, 20, 20, 40, 40, 20] * 2,
                dtype=np.double,
            )

            kds = np.array(
                [1.8, 0.8, 0.8, 1.8, 1.8, 0.6] * 2,
                dtype=np.double,
            )

            tau_limit = (
                40.0
                * np.ones(
                    12,
                    dtype=np.double,
                )
            )

    if not glfw.init():
        raise RuntimeError(
            "Could not initialize GLFW"
        )

    try:
        policy = torch.jit.load(
            args.load_model
        )

        policy.eval()

        visualizer = mujoco_visual(
            target_x=args.target_x,
            target_y=args.target_y,
            max_vx=args.max_vx,
            max_wz=args.max_wz,
        )

        visualizer.run_mujoco(
            policy,
            Sim2simCfg(),
        )

    finally:
        glfw.terminate()

    print("Simulation completed.")