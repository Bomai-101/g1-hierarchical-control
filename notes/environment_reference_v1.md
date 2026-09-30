# Environment Reference V1 — Phase 0 MuJoCo sanity audit

Status: **configuration accepted for the next diagnostic phase**. This is a
simulation reference, not validation against physical G1 hardware. No physics,
pose, PD, action, reward, or termination parameter was changed.

## Reproduce

From the project root, in `.venv`:

```bash
python3 scripts/audit_mujoco_environment.py
```

The detailed machine-readable output is local-only at
`code/day9/g1_balance/checkpoints/environment_audits/phase0_environment_audit.json`.
The checkpoint tree is already ignored by Git. The audit script recreates a
fresh `G1Env`, lets its existing 25 physics-step PD reset settling finish, and
then applies the unchanged zero residual action until termination.

## Exact model identity

- MuJoCo Python runtime: `3.14.0`.
- Unitree source checkout: `unitreerobotics/unitree_mujoco`, commit
  `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` (2026-09-07).
- Loaded scene: `~/robotics/unitree_mujoco/unitree_robots/g1/scene_29dof.xml`.
- Scene SHA-256: `e254f11acce2ec6f6efa5bf9b15e288bbd0ca29aeeabb1e1d0fea92f65436bbf`.
- Included robot XML SHA-256:
  `423e28bd718b19f7a65cda539b6f794ddbb268b4b9bdbd85f4bd982b30729617`.
- Both XML files were clean in the Unitree checkout at audit time.
- Compiled dimensions: `nq=36`, `nv=35`, `nu=29`, `ngeom=74`.

## Physics and contacts

- Runtime timestep `0.002 s`; 10 physics steps per policy step (`0.020 s`, 50 Hz).
- Integrator Euler; solver Newton; 100 iterations; tolerance `1e-8`.
- Gravity `(0, 0, -9.81) m/s²`; disable flags `0`; enable flags `0`.
- Floor is a collidable plane (`contype=1`, `conaffinity=1`, `condim=3`).
- Each foot has four collidable sphere geoms on its ankle-roll body. All eight
  floor/foot pairs pass the compiled collision mask. The visible foot mesh
  itself is not the colliding geom.
- Compiled floor and foot geom friction: `(1, 0.005, 0.0001)`;
  `condim=3` supports tangential friction. Neither XML has an explicit
  `<contact><pair>` override.
- No XML equality/weld, mocap body, or tendon support. Compiled model has
  `neq=0`, `ntendon=0`, `nmocap=0`; applied external forces were zero at reset.

## Passive joints and actuation

- The twelve leg joints use damping `0.05`, armature `0.01`, friction loss
  `0.2`. Their compiled position limits match the joint ranges in the Unitree
  XML; all leg joints are limit-enabled.
- Hip motor control ranges are `±88`, knee `±139`, ankle `±50`.
  MuJoCo actuator `ctrllimited=True`; motor-level `forcelimited=False`.
  **Joint-level** actuator force limiting is enabled for all twelve leg joints,
  with matching `±88/±139/±50` ranges. These are independent mechanisms.
- The existing controller clips raw PD torque to each actuator `ctrlrange`
  before setting `data.ctrl`. Along this zero-policy trajectory, **0 of
  62,350 actuator-physics commands were clipped** and none reached 95% of
  their configured magnitude. Maximum raw torques on the controlled legs:
  hip pitch about `8.14`, knee `11.13`, ankle pitch `20.24` (same units as
  motor control/torque, respectively), all below `88/139/50`.
- This only establishes headroom along the nominal zero-policy trajectory;
  a later active recovery action can still saturate and must be measured
  separately.

## Dynamic zero-policy check

- Baseline reproduced exactly: **215 policy steps = 4.30 s after settling**;
  terminated by the existing fall rule.
- Both feet had floor contact for `2149/2150` sampled physics steps after
  settling; contact counts were left `8397`, right `8396` across the four
  contact points per foot.
- Horizontal velocity at contacting foot material points: median about
  `0.00020 m/s`, 95th percentile about `0.0040 m/s`, maximum about
  `0.0205 m/s` on each side. Tangential/normal contact-force ratio: median
  `0.007`, 95th percentile `0.145`, maximum `0.778`.
- At policy step 100: pitch `-0.0044 rad`, pitch rate `+0.0063 rad/s`.
  At 125: `+0.0011`, `+0.0181`; at 150: `+0.0186`, `+0.0626`;
  at 175: `+0.0874`, `+0.2566`; at 200: `+0.3675`, `+1.0169`.
  At termination step 215: pitch `+0.829 rad`, height `0.644 m`.

The nominal failure is **consistent with progressive forward toppling**.
These measurements do not suggest gross foot sliding or actuator clipping as
the primary failure in this one run. They do not prove that a change to
friction, pose, contact model, or control policy could not improve standing,
and they do not establish physical fidelity to a real G1.

## Gate decision

The loaded model is a clean local checkout of Unitree's 29DOF MuJoCo model;
its runtime physics and contact settings are coherent, both feet actually
contact the floor, no artificial support is active, and the historical zero
baseline reproduces. **Freeze this exact configuration as Environment
Reference V1 and proceed to full zero/PPO trajectory logging.** If the scene,
robot XML, MuJoCo version, or environment/control settings change, repeat this
audit and remeasure the 215/241-step references before comparing outcomes.

Reference model and semantics:

- [Unitree 29DOF MuJoCo model](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/g1_29dof.xml)
- [Unitree G1 29DOF scene](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/scene_29dof.xml)
- [MuJoCo contact model](https://mujoco.readthedocs.io/en/latest/computation/)
- [MuJoCo XML defaults](https://mujoco.readthedocs.io/en/latest/XMLreference.html)
