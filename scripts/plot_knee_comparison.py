import pandas as pd
import matplotlib.pyplot as plt

df20 = pd.read_csv("knee_tracking_kp20_kd2.csv")
df40 = pd.read_csv("knee_tracking_kp40_kd2.csv")

# Relative displacement
delta_q20 = df20["current_q"] - df20["current_q"].iloc[0]
delta_q40 = df40["current_q"] - df40["current_q"].iloc[0]

target20 = df20["target_q"] - df20["current_q"].iloc[0]
target40 = df40["target_q"] - df40["current_q"].iloc[0]

fig, axs = plt.subplots(1, 2, figsize=(14, 5))


# ============================================================
# Left: Relative Position Tracking
# ============================================================

axs[0].plot(
    df20["time"],
    target20,
    label="Target (+0.1 rad)"
)

axs[0].plot(
    df20["time"],
    delta_q20,
    label="Kp=20, Kd=2"
)

axs[0].plot(
    df40["time"],
    delta_q40,
    label="Kp=40, Kd=2"
)

axs[0].set_title("G1 Left Knee Relative Position Tracking")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Relative Joint Position Δq (rad)")
axs[0].grid(True)
axs[0].legend()


# ============================================================
# Right: Tracking Error
# ============================================================

axs[1].plot(
    df20["time"],
    df20["error"],
    label="Kp=20, Kd=2"
)

axs[1].plot(
    df40["time"],
    df40["error"],
    label="Kp=40, Kd=2"
)

axs[1].set_title("G1 Left Knee Tracking Error Comparison")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Tracking Error (rad)")
axs[1].grid(True)
axs[1].legend()


plt.tight_layout()
plt.savefig("knee_tracking_normalized_comparison.png", dpi=300)
plt.close()

print("Saved: knee_tracking_normalized_comparison.png")
