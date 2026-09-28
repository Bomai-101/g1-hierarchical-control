from pathlib import Path

import mujoco


def find_g1_scene():

    g1_dir = (
        Path.home()
        / "robotics"
        / "unitree_mujoco"
        / "unitree_robots"
        / "g1"
    )

    candidates = [
        g1_dir / "scene_29dof.xml",
        g1_dir / "scene.xml",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find a G1 scene in {g1_dir}"
    )


def main():

    scene_path = find_g1_scene()

    print("Loading:")
    print(scene_path)
    print()

    model = mujoco.MjModel.from_xml_path(
        str(scene_path)
    )

    data = mujoco.MjData(model)

    # Make MuJoCo compute all derived state once
    mujoco.mj_forward(model, data)

    print("=== MODEL DIMENSIONS ===")
    print("nq =", model.nq)
    print("nv =", model.nv)
    print("nu =", model.nu)
    print("timestep =", model.opt.timestep)
    print()

    print("=== JOINTS ===")

    for i in range(model.njnt):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            i
        )

        qpos_adr = model.jnt_qposadr[i]
        dof_adr = model.jnt_dofadr[i]

        print(
            f"{i:2d} | "
            f"{str(name):30s} | "
            f"qpos={qpos_adr:2d} | "
            f"qvel={dof_adr:2d}"
        )

    print()

    print("=== ACTUATORS ===")

    for i in range(model.nu):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i
        )

        print(
            f"{i:2d} | {name}"
        )

    print()

    print("=== RAW STATE SHAPES ===")

    print(
        "qpos shape:",
        data.qpos.shape
    )

    print(
        "qvel shape:",
        data.qvel.shape
    )

    print(
        "ctrl shape:",
        data.ctrl.shape
    )

    print()

    print("First qpos values:")
    print(data.qpos[:10])

    print()

    print("First qvel values:")
    print(data.qvel[:10])


if __name__ == "__main__":
    main()