from pathlib import Path
import math

import mujoco
import numpy as np
from g1_control.config import balance as config



class G1Env:

    # ========================================================
    # Dimensions
    # ========================================================

    OBS_DIM = config.OBS_DIM

    # Real G1 actuated joints
    NUM_JOINTS = config.NUM_JOINTS

    # PPO now controls only 6 balance-critical joints
    ACTION_DIM = config.ACTION_DIM

    # Mapping:
    #
    # PPO action index
    #       ↓
    # G1 joint index
    #
    # 0 -> left hip pitch
    # 1 -> left knee
    # 2 -> left ankle pitch
    # 3 -> right hip pitch
    # 4 -> right knee
    # 5 -> right ankle pitch

    POLICY_JOINT_INDICES = np.array(
        [
            0,
            3,
            4,
            6,
            9,
            10
        ],
        dtype=np.int64
    )

    def __init__(self):

        # ====================================================
        # Load G1 model
        # ====================================================

        g1_dir = (
            Path.home()
            / "robotics"
            / "unitree_mujoco"
            / "unitree_robots"
            / "g1"
        )

        scene_path = (
            g1_dir
            / "scene_29dof.xml"
        )

        if not scene_path.exists():

            raise FileNotFoundError(
                f"G1 scene not found: "
                f"{scene_path}"
            )

        self.model = (
            mujoco.MjModel.from_xml_path(
                str(scene_path)
            )
        )

        self.data = mujoco.MjData(
            self.model
        )

        # ====================================================
        # Standing reference pose
        # ====================================================

        self.default_q = np.zeros(
            self.NUM_JOINTS,
            dtype=np.float32
        )

        # Left leg
        self.default_q[0:6] = [
            config.HIP_PITCH,   # hip pitch
             0.0,   # hip roll
             0.0,   # hip yaw
            config.KNEE,   # knee
            config.ANKLE_PITCH,   # ankle pitch
             0.0    # ankle roll
        ]

        # Right leg
        self.default_q[6:12] = [
            config.HIP_PITCH,
            0.0,
            0.0,
            config.KNEE,
            config.ANKLE_PITCH,
            0.0
        ]

        # ====================================================
        # Action configuration
        # ====================================================

        # Day 8:
        # 0.10 rad
        #
        # Day 9:
        # slightly larger authority,
        # but only on six important joints.
        self.action_scale = config.ACTION_SCALE

        # ====================================================
        # PD gains
        # ====================================================

        self.kp = np.full(
            self.NUM_JOINTS,
            40.0,
            dtype=np.float32
        )

        self.kd = np.full(
            self.NUM_JOINTS,
            1.0,
            dtype=np.float32
        )

        # Lower-body gains
        self.kp[0:12] = np.array(
            config.LOWER_BODY_KP,
            dtype=np.float32
        )

        self.kd[0:12] = np.array(
            config.LOWER_BODY_KD,
            dtype=np.float32
        )

        # ====================================================
        # Actuator torque limits
        # ====================================================

        self.torque_limits = np.array(
            self.model.actuator_ctrlrange,
            dtype=np.float32
        )

        # ====================================================
        # Frequency
        # ====================================================

        # MuJoCo:
        # dt = 0.002 s → 500 Hz
        #
        # PPO:
        # every 10 physics steps → 50 Hz

        self.decimation = 10

        # ====================================================
        # Episode configuration
        # ====================================================

        self.max_episode_steps = 1000

        self.episode_step = 0

        # Short PD-only settling phase
        # during reset.
        #
        # 25 * 0.002 = 0.05 s
        self.settling_physics_steps = 25

        self.reference_height = 0.80

        self.reset()

    # ========================================================
    # Quaternion → RPY
    # ========================================================

    def quaternion_to_rpy(
        self,
        quat
    ):

        w, x, y, z = quat

        # Roll
        sinr_cosp = 2.0 * (
            w * x
            + y * z
        )

        cosr_cosp = (
            1.0
            - 2.0
            * (
                x * x
                + y * y
            )
        )

        roll = math.atan2(
            sinr_cosp,
            cosr_cosp
        )

        # Pitch
        sinp = 2.0 * (
            w * y
            - z * x
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
            w * z
            + x * y
        )

        cosy_cosp = (
            1.0
            - 2.0
            * (
                y * y
                + z * z
            )
        )

        yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        return np.array(
            [
                roll,
                pitch,
                yaw
            ],
            dtype=np.float32
        )

    # ========================================================
    # Joint state
    # ========================================================

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

    # ========================================================
    # 64D observation
    # ========================================================

    def get_observation(self):

        q, dq = (
            self.get_joint_state()
        )

        quat = (
            self.data.qpos[3:7]
        )

        rpy = (
            self.quaternion_to_rpy(
                quat
            )
        )

        gyro = np.array(
            self.data.qvel[3:6],
            dtype=np.float32
        )

        observation = np.concatenate(
            [
                q,
                dq,
                rpy,
                gyro
            ]
        )

        assert observation.shape == (
            self.OBS_DIM,
        )

        return observation.astype(
            np.float32
        )

    # ========================================================
    # 6D policy action → 29D target q
    # ========================================================

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
        ), (
            f"Expected action shape "
            f"({self.ACTION_DIM},), "
            f"got {action.shape}"
        )

        # Tanh should already produce
        # values inside (-1, 1).
        #
        # This remains a defensive guard.
        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # Start from complete
        # 29-joint standing pose.
        target_q = (
            self.default_q.copy()
        )

        # Only six selected joints
        # receive PPO offsets.
        target_q[
            self.POLICY_JOINT_INDICES
        ] += (
            self.action_scale
            * action
        )

        return target_q

    # ========================================================
    # Low-level PD
    # ========================================================

    def compute_pd_torque(
        self,
        target_q
    ):

        q, dq = (
            self.get_joint_state()
        )

        torque = (
            self.kp
            * (
                target_q - q
            )
            - self.kd
            * dq
        )

        torque = np.clip(
            torque,
            self.torque_limits[:, 0],
            self.torque_limits[:, 1]
        )

        return torque

    # ========================================================
    # Day 9 balance reward
    # ========================================================

    def compute_reward(
        self,
        action
    ):

        obs = self.get_observation()

        dq = obs[29:58]

        roll = float(
            obs[58]
        )

        pitch = float(
            obs[59]
        )

        gyro = obs[61:64]

        height = float(
            self.data.qpos[2]
        )

        # ----------------------------------------------------
        # 1. Upright reward
        #
        # Day 8 used a relatively weak quadratic signal.
        #
        # Now even moderate tilt produces a visible
        # reward difference.
        # ----------------------------------------------------

        upright_reward = np.exp(
            -4.0
            * (
                roll ** 2
                + pitch ** 2
            )
        )

        # ----------------------------------------------------
        # 2. Height reward
        # ----------------------------------------------------

        height_error = (
            height
            - self.reference_height
        )

        height_reward = np.exp(
            -20.0
            * (
                height_error ** 2
            )
        )

        # ----------------------------------------------------
        # 3. Angular velocity penalty
        #
        # Penalize a robot that is rotating/falling
        # quickly even before the orientation is large.
        # ----------------------------------------------------

        angular_velocity_penalty = (
            np.mean(
                gyro ** 2
            )
        )

        # ----------------------------------------------------
        # 4. Selected joint velocity penalty
        # ----------------------------------------------------

        selected_dq = dq[
            self.POLICY_JOINT_INDICES
        ]

        joint_velocity_penalty = (
            np.mean(
                selected_dq ** 2
            )
        )

        # ----------------------------------------------------
        # 5. Action penalty
        # ----------------------------------------------------

        action_penalty = (
            np.mean(
                action ** 2
            )
        )

        # ----------------------------------------------------
        # Total reward
        # ----------------------------------------------------

        reward = (
            0.2
            + 1.5
            * upright_reward
            + 0.5
            * height_reward
            - 0.02
            * angular_velocity_penalty
            - 0.002
            * joint_velocity_penalty
            - 0.005
            * action_penalty
        )

        return float(
            reward
        )

    # ========================================================
    # Episode termination
    # ========================================================

    def check_termination(self):

        obs = self.get_observation()

        roll = float(
            obs[58]
        )

        pitch = float(
            obs[59]
        )

        height = float(
            self.data.qpos[2]
        )

        fallen = (
            height < 0.45
            or abs(roll) > 0.8
            or abs(pitch) > 0.8
        )

        timeout = (
            self.episode_step
            >= self.max_episode_steps
        )

        return (
            fallen,
            timeout
        )

    # ========================================================
    # RL step
    # ========================================================

    def step(
        self,
        action
    ):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        target_q = (
            self.action_to_target_q(
                action
            )
        )

        # ----------------------------------------------------
        # Hold the same high-level target
        # for 10 physics steps.
        # ----------------------------------------------------

        for _ in range(
            self.decimation
        ):

            torque = (
                self.compute_pd_torque(
                    target_q
                )
            )

            self.data.ctrl[:] = (
                torque
            )

            mujoco.mj_step(
                self.model,
                self.data
            )

        self.episode_step += 1

        next_obs = (
            self.get_observation()
        )

        reward = (
            self.compute_reward(
                action
            )
        )

        (
            terminated,
            truncated
        ) = self.check_termination()

        # ----------------------------------------------------
        # Explicit fall penalty
        # ----------------------------------------------------

        if terminated:

            reward -= 5.0

        return (
            next_obs,
            reward,
            terminated,
            truncated
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # ----------------------------------------------------
        # Floating base
        # ----------------------------------------------------

        self.data.qpos[0:3] = [
            0.0,
            0.0,
            0.80
        ]

        self.data.qpos[3:7] = [
            1.0,
            0.0,
            0.0,
            0.0
        ]

        # ----------------------------------------------------
        # Standing pose
        # ----------------------------------------------------

        self.data.qpos[7:36] = (
            self.default_q
        )

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        # ----------------------------------------------------
        # Short PD settling period
        #
        # PPO receives no reward and takes no action here.
        # ----------------------------------------------------

        for _ in range(
            self.settling_physics_steps
        ):

            torque = (
                self.compute_pd_torque(
                    self.default_q
                )
            )

            self.data.ctrl[:] = (
                torque
            )

            mujoco.mj_step(
                self.model,
                self.data
            )

        # Episode starts AFTER settling.
        self.episode_step = 0

        # Use settled height as reference.
        self.reference_height = float(
            self.data.qpos[2]
        )

        return self.get_observation()
