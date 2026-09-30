"""Single source of truth for the frozen Day 9 balance experiment."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Keep experiment artifacts in their Day 9 location.  Package code is reusable;
# recorded data remains part of the historical experiment.
BASE_DIR = PROJECT_ROOT / "code" / "day9" / "g1_balance"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RUNS_DIR = CHECKPOINT_DIR / "runs"

SEED = 42

OBS_DIM = 64
ACTION_DIM = 6
NUM_JOINTS = 29
POLICY_JOINT_INDICES = (0, 3, 4, 6, 9, 10)

# Frozen standing pose and low-level control reference.
HIP_PITCH = -0.16
KNEE = 0.23
ANKLE_PITCH = -0.07
ACTION_SCALE = 0.15

LOWER_BODY_KP = (100.0, 100.0, 100.0, 150.0, 40.0, 40.0) * 2
LOWER_BODY_KD = (2.0, 2.0, 2.0, 4.0, 2.0, 2.0) * 2

# Frozen PPO pilot configuration.
ROLLOUT_STEPS = 2048
SMOKE_ROLLOUT_STEPS = 512
NUM_UPDATES = 30
PPO_EPOCHS = 3
BATCH_SIZE = 128
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ACTOR_LR = 4e-5
CRITIC_LR = 3e-4
ENTROPY_COEF = 0.0
ACTOR_LOG_STD = -3.0
CHECKPOINT_EVERY = 1

# Stage B reset curriculum. Evaluation remains deterministic.
RESET_NOMINAL_PROBABILITY = 0.30
RESET_PITCH_RANGE = (-0.005, 0.005)
RESET_PITCH_RATE_RANGE = (-0.05, 0.05)

BEST_DETERMINISTIC_CHECKPOINT = "best_deterministic.pt"
