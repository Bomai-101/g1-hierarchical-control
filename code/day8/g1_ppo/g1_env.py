from pathlib import Path
import math

import mujoco
import numpy as np


class G1Env:

    OBS_DIM = 64
    ACTION_DIM = 29

    def __init__(self):

        g1_dir = (
            Path.home()
            / "robotics"
            / "unitree_mujoco"
            / "unitree_robots"
            / "g1"
        )

        scene_path = g1_dir / "scene_29dof.xml"

        if not scene_path.exists():
            raise FileNotFoundError(
                f"G1 scene not found: {scene_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(
            str(scene_path)
        )

        self.data = mujoco.MjData(
            self.model
        )

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.initial_height = 0.80

        self.max_episode_steps = 1000
        self.episode_step = 0

        # Initial standing joint pose
        self.default_q = np.zeros(
            self.ACTION_DIM,
            dtype=np.float32
        )

        # Left leg
        self.default_q[0:6] = [
            -0.1,   # hip pitch
            0.0,   # hip roll
            0.0,   # hip yaw
            0.3,   # knee
            -0.2,   # ankle pitch
            0.0    # ankle roll
        ]

        # Right leg
        self.default_q[6:12] = [
            -0.1,
            0.0,
            0.0,
            0.3,
            -0.2,
            0.0
        ]

        # waist + arms remain zero for now

        # PPO action scaling:
        # normalized policy action → joint position offset
        self.action_scale = 0.10

        self.kp = np.full(
            self.ACTION_DIM,
            40.0,
            dtype=np.float32
        )

        self.kd = np.full(
            self.ACTION_DIM,
            1.0,
            dtype=np.float32
        )

        # Unitree-style leg gains
        self.kp[0:12] = np.array([
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40
        ], dtype=np.float32)

        self.kd[0:12] = np.array([
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2
        ], dtype=np.float32)

        # MuJoCo actuator torque limits
        self.torque_limits = np.array(
            self.model.actuator_ctrlrange,
            dtype=np.float32
        )

        # 500 Hz physics, 50 Hz policy
        self.decimation = 10

        self.reset()

    def quaternion_to_rpy(self, quat):

        # MuJoCo free-joint quaternion:
        # [w, x, y, z]

        w, x, y, z = quat

        # Roll
        sinr_cosp = 2.0 * (
            w * x + y * z
        )

        cosr_cosp = 1.0 - 2.0 * (
            x * x + y * y
        )

        roll = math.atan2(
            sinr_cosp,
            cosr_cosp
        )

        # Pitch
        sinp = 2.0 * (
            w * y - z * x
        )

        sinp = np.clip(
            sinp,
            -1.0,
            1.0
        )

        pitch = math.asin(
            sinp
        )

        # Yaw
        siny_cosp = 2.0 * (
            w * z + x * y
        )

        cosy_cosp = 1.0 - 2.0 * (
            y * y + z * z
        )

        yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        return np.array(
            [roll, pitch, yaw],
            dtype=np.float32
        )

    def get_observation(self):

        # -------------------------
        # 1. Joint positions
        # -------------------------

        q = np.array(
            self.data.qpos[7:36],
            dtype=np.float32
        )

        # -------------------------
        # 2. Joint velocities
        # -------------------------

        dq = np.array(
            self.data.qvel[6:35],
            dtype=np.float32
        )

        # -------------------------
        # 3. Base orientation
        # -------------------------

        quat = self.data.qpos[3:7]

        rpy = self.quaternion_to_rpy(
            quat
        )

        # -------------------------
        # 4. Base angular velocity
        # -------------------------

        gyro = np.array(
            self.data.qvel[3:6],
            dtype=np.float32
        )

        # -------------------------
        # 5. Build observation
        # -------------------------

        observation = np.concatenate([
            q,
            dq,
            rpy,
            gyro
        ])

        assert observation.shape == (
            self.OBS_DIM,
        ), (
            f"Expected observation "
            f"shape ({self.OBS_DIM},), "
            f"got {observation.shape}"
        )

        return observation.astype(
            np.float32
        )

    def get_joint_state(self):

        q = np.array(
            self.data.qpos[7:36],
            dtype=np.float32
        )

        dq = np.array(
            self.data.qvel[6:35],
            dtype=np.float32
        )

        return q, dq

    def compute_pd_torque(
        self,
        target_q
    ):

        q, dq = self.get_joint_state()

        torque = (
            self.kp * (target_q - q)
            - self.kd * dq
        )

        torque = np.clip(
            torque,
            self.torque_limits[:, 0],
            self.torque_limits[:, 1]
        )

        return torque

    def action_to_target_q(
        self,
        action
    ):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        assert action.shape == (
            self.ACTION_DIM,
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        target_q = (
            self.default_q
            + self.action_scale * action
        )

        return target_q

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        target_q = self.action_to_target_q(
            action
        )

        for _ in range(self.decimation):

            torque = self.compute_pd_torque(
                target_q
            )

            self.data.ctrl[:] = torque

            mujoco.mj_step(
                self.model,
                self.data
            )

        self.episode_step += 1

        next_obs = self.get_observation()

        reward = self.compute_reward(
            action
        )

        terminated, truncated = (
            self.check_termination()
        )

        return (
            next_obs,
            reward,
            terminated,
            truncated
        )

    def reset(self):

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # Floating base position
        self.data.qpos[0:3] = [
            0.0,
            0.0,
            0.80
        ]

        # Identity quaternion
        self.data.qpos[3:7] = [
            1.0,
            0.0,
            0.0,
            0.0
        ]

        # Standing reference pose
        self.data.qpos[7:36] = (
            self.default_q
        )

        # Start with zero velocity
        self.data.qvel[:] = 0.0

        # Start with zero actuator torque
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.episode_step = 0

        return self.get_observation()

    def compute_reward(
        self,
        action
    ):

        obs = self.get_observation()

        q = obs[0:29]
        dq = obs[29:58]

        roll, pitch, yaw = obs[58:61]

        height = self.data.qpos[2]

        # -------------------------
        # Upright reward
        # -------------------------

        upright_reward = (
            1.0
            - roll ** 2
            - pitch ** 2
        )

        # -------------------------
        # Height reward
        # -------------------------

        height_error = (
            height
            - self.initial_height
        )

        height_reward = -(
            height_error ** 2
        )

        # -------------------------
        # Joint velocity penalty
        # -------------------------

        velocity_penalty = (
            np.mean(dq ** 2)
        )

        # -------------------------
        # Action penalty
        # -------------------------

        action_penalty = (
            np.mean(action ** 2)
        )

        reward = (
            1.0
            + 1.0 * upright_reward
            + 0.5 * height_reward
            - 0.01 * velocity_penalty
            - 0.01 * action_penalty
        )

        return float(reward)

    def check_termination(self):

        obs = self.get_observation()

        roll, pitch, _ = obs[58:61]

        height = self.data.qpos[2]

        fallen = (
            height < 0.45
            or abs(roll) > 0.8
            or abs(pitch) > 0.8
        )

        timeout = (
            self.episode_step
            >= self.max_episode_steps
        )

        return fallen, timeout

if __name__ == "__main__":

    env = G1Env()

    obs = env.reset()

    zero_action = np.zeros(
        env.ACTION_DIM,
        dtype=np.float32
    )

    print("=== ZERO ACTION RL ENV TEST ===")

    print(
        "Initial height:",
        env.data.qpos[2]
    )

    print(
        "Initial RPY:",
        obs[58:61]
    )

    print()

    for i in range(100):

        (
            obs,
            reward,
            terminated,
            truncated
        ) = env.step(
            zero_action
        )

        if i % 10 == 0:

            print(
                f"step={i:3d} | "
                f"height={env.data.qpos[2]:.3f} | "
                f"pitch={obs[59]:.3f} | "
                f"reward={reward:.3f} | "
                f"terminated={terminated} | "
                f"truncated={truncated}"
            )

        if terminated or truncated:

            print()

            print(
                "Episode ended at step:",
                i
            )

            print(
                "Reason:",
                "terminated"
                if terminated
                else "truncated"
            )

            break