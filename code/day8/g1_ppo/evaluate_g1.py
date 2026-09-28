from pathlib import Path
import argparse
import time

import mujoco.viewer
import numpy as np
import torch

from g1_env import G1Env
from networks import Actor


OBS_DIM = 64
ACTION_DIM = 29
MAX_STEPS = 1000


# ============================================================
# Create untrained actor
# ============================================================

def create_untrained_actor(device):

    # Fixed seed so that this baseline
    # is reproducible every time.
    torch.manual_seed(42)

    actor = Actor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM
    ).to(device)

    actor.eval()

    return actor


# ============================================================
# Load trained actor
# ============================================================

def load_trained_actor(
    device,
    checkpoint_path
):

    actor = Actor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    actor.load_state_dict(
        checkpoint["actor"]
    )

    actor.eval()

    print(
        "Loaded checkpoint:",
        checkpoint_path
    )

    if "update" in checkpoint:
        print(
            "Checkpoint update:",
            checkpoint["update"]
        )

    if "total_env_steps" in checkpoint:
        print(
            "Checkpoint env steps:",
            checkpoint["total_env_steps"]
        )

    if "mean_episode_length" in checkpoint:
        print(
            "Saved mean episode length:",
            checkpoint[
                "mean_episode_length"
            ]
        )

    if "mean_episode_reward" in checkpoint:
        print(
            "Saved mean episode reward:",
            checkpoint[
                "mean_episode_reward"
            ]
        )

    print()

    return actor


# ============================================================
# Get action
# ============================================================

def get_action(
    policy_name,
    actor,
    obs,
    device
):

    # --------------------------------------------------------
    # Zero-action PD baseline
    #
    # action = 0
    # therefore:
    #
    # q_target = default_q
    # --------------------------------------------------------

    if policy_name == "zero":

        return np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Actor policy
    # --------------------------------------------------------

    obs_tensor = (
        torch.from_numpy(obs)
        .float()
        .to(device)
    )

    with torch.no_grad():

        dist = actor(
            obs_tensor
        )

        # Deterministic evaluation:
        #
        # Training:
        # raw_action = dist.sample()
        #
        # Evaluation:
        # raw_action = dist.mean
        raw_action = dist.mean

        # Same bounded-action mapping
        # used during training.
        action = torch.tanh(
            raw_action
        )

    return (
        action
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    policy_name,
    actor,
    device,
    use_viewer=False
):

    env = G1Env()

    obs = env.reset()

    total_reward = 0.0

    max_abs_pitch = 0.0

    min_height = float(
        env.data.qpos[2]
    )

    terminated = False
    truncated = False

    # --------------------------------------------------------
    # 500 Hz physics / decimation 10
    #
    # policy_dt = 0.002 * 10 = 0.02 s
    # = 50 Hz policy
    # --------------------------------------------------------

    policy_dt = (
        env.model.opt.timestep
        * env.decimation
    )

    viewer = None

    # --------------------------------------------------------
    # Launch viewer if requested
    # --------------------------------------------------------

    if use_viewer:

        viewer = (
            mujoco.viewer.launch_passive(
                env.model,
                env.data
            )
        )

        print()
        print(
            "MuJoCo viewer started."
        )

        print(
            "When the episode finishes, "
            "close the viewer window using X."
        )

        print(
            "Avoid Ctrl+C / Ctrl+Z while "
            "the viewer is running."
        )

        print()

    try:

        for step in range(
            MAX_STEPS
        ):

            # ------------------------------------------------
            # Viewer may have been manually closed
            # ------------------------------------------------

            if (
                viewer is not None
                and not viewer.is_running()
            ):

                print(
                    "Viewer closed."
                )

                break

            step_start = (
                time.time()
            )

            # =================================================
            # Policy
            # =================================================

            action = get_action(
                policy_name,
                actor,
                obs,
                device
            )

            # =================================================
            # Environment step
            # =================================================

            (
                next_obs,
                reward,
                terminated,
                truncated
            ) = env.step(
                action
            )

            total_reward += reward

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            pitch = float(
                next_obs[59]
            )

            height = float(
                env.data.qpos[2]
            )

            max_abs_pitch = max(
                max_abs_pitch,
                abs(pitch)
            )

            min_height = min(
                min_height,
                height
            )

            obs = next_obs

            # =================================================
            # Viewer sync
            # =================================================

            if viewer is not None:

                viewer.sync()

            # =================================================
            # Episode boundary
            # =================================================

            if (
                terminated
                or truncated
            ):

                if viewer is not None:

                    print()
                    print(
                        "Episode finished."
                    )

                    print(
                        f"Policy: {policy_name}"
                    )

                    print(
                        "Close the MuJoCo "
                        "viewer window using X "
                        "to return to terminal."
                    )

                    # Keep the final frame visible.
                    #
                    # Instead of immediately calling
                    # viewer.close(), wait for the user
                    # to close the native window.
                    while viewer.is_running():

                        viewer.sync()

                        time.sleep(
                            0.05
                        )

                break

            # =================================================
            # Keep viewer approximately real time
            # =================================================

            if viewer is not None:

                elapsed = (
                    time.time()
                    - step_start
                )

                remaining = (
                    policy_dt
                    - elapsed
                )

                if remaining > 0:

                    time.sleep(
                        remaining
                    )

    finally:

        # ----------------------------------------------------
        # Normally, if the user already closed the viewer
        # window, viewer.is_running() will be False.
        #
        # This is only a final cleanup fallback.
        # ----------------------------------------------------

        if (
            viewer is not None
            and viewer.is_running()
        ):

            try:

                viewer.close()

            except Exception as exc:

                print(
                    "Viewer cleanup warning:",
                    exc
                )

    # --------------------------------------------------------
    # step exists after normal execution of loop.
    # --------------------------------------------------------

    episode_length = (
        step + 1
    )

    result = {
        "policy":
            policy_name,

        "episode_length":
            episode_length,

        "return":
            total_reward,

        "max_abs_pitch":
            max_abs_pitch,

        "min_height":
            min_height,

        "terminated":
            terminated,

        "truncated":
            truncated,
    }

    return result


# ============================================================
# Print result
# ============================================================

def print_result(
    result
):

    print(
        f"{result['policy']:10s} | "
        f"length={result['episode_length']:4d} | "
        f"return={result['return']:8.2f} | "
        f"max_pitch={result['max_abs_pitch']:.3f} | "
        f"min_height={result['min_height']:.3f} | "
        f"terminated={result['terminated']} | "
        f"truncated={result['truncated']}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy",
        choices=[
            "all",
            "zero",
            "untrained",
            "trained"
        ],
        default="all"
    )

    parser.add_argument(
        "--viewer",
        action="store_true"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Viewer mode should evaluate only one policy at a time.
    # --------------------------------------------------------

    if (
        args.policy == "all"
        and args.viewer
    ):

        raise ValueError(
            "Use --viewer with one policy "
            "at a time."
        )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Evaluation device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # Checkpoint path
    # ========================================================

    checkpoint_path = (
        Path(__file__).parent
        / "checkpoints"
        / "best.pt"
    )

    # ========================================================
    # Headless comparison
    # ========================================================

    if args.policy == "all":

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        untrained_actor = (
            create_untrained_actor(
                device
            )
        )

        trained_actor = (
            load_trained_actor(
                device,
                checkpoint_path
            )
        )

        print(
            "=== HEADLESS COMPARISON ==="
        )

        policies = [
            (
                "zero",
                None
            ),
            (
                "untrained",
                untrained_actor
            ),
            (
                "trained",
                trained_actor
            ),
        ]

        for (
            policy_name,
            actor
        ) in policies:

            result = evaluate_episode(
                policy_name,
                actor,
                device,
                use_viewer=False
            )

            print_result(
                result
            )

        return

    # ========================================================
    # Single-policy evaluation
    # ========================================================

    if args.policy == "zero":

        # No neural network is required
        # for the zero-action PD baseline.
        actor = None

    elif (
        args.policy
        == "untrained"
    ):

        actor = (
            create_untrained_actor(
                device
            )
        )

    elif (
        args.policy
        == "trained"
    ):

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        actor = (
            load_trained_actor(
                device,
                checkpoint_path
            )
        )

    else:

        raise ValueError(
            f"Unknown policy: "
            f"{args.policy}"
        )

    # ========================================================
    # Evaluate
    # ========================================================

    result = evaluate_episode(
        args.policy,
        actor,
        device,
        use_viewer=args.viewer
    )

    print()
    print_result(
        result
    )


if __name__ == "__main__":
    main()