# 同指令 Isaac / MuJoCo 动画比较

打开 [回放页面](index.html)。默认展示续训后 model_1999，可切换续训前 model_1499。
页面支持同步播放、暂停、拖动时间和 0.5× / 0.25× 慢放，下载单独或并排 MP4。
本地服务地址：<http://localhost:8765/visual_comparison/>。服务停止后仍可直接打开 HTML，须保留同目录视频。

## 指令与观察

策略输入的速度指令固定为 `obs[9:12] = [1, 0, 0]`：前进 1 m/s、横向 0 m/s、转向角速度 0 rad/s。
本例初始航向为 0，关闭 heading feedback；不要求额外的航向纠偏。指令由任务命令项进入 123D 策略观测，策略输出 37D 关节动作。
对每次动作输入均检查指令相同，不更改策略、执行器增益或监测/切换逻辑。

| 模型 | Isaac 展示环境 | MuJoCo | 展示时长 |
|---|---|---|---|
| 1499 | 完成 8 秒 | 完成 8 秒，方向仍有偏差 | 8 秒 |
| 1999 | 完成 8 秒 | 2.60 秒跌倒终止 | 8 秒；终止后明确标红冻结 |

这是一条固定前进指令的单次演示，不能替代转向或鲁棒性验收。1999 未通过先前的迁移评估；1499 仅保留为实验对照。完整 48 组结果见 [续训评估](../posttraining/README.md)。

## 画面来源与限制

左侧为 **Isaac GPU 物理计算得到的真实身体姿态离线回放**，不是 Isaac 原生窗口录像。
当前 WSL 的 Vulkan 设备创建失败，不能录制 Isaac 原生画面；物理仍运行在 CUDA 上。
提取 Isaac 原始 `g1_minimal.usd` 的 44 个刚体的视觉网格，逐帧应用 Isaac 记录的各部件世界位置与四元数。
仅以 MuJoCo 的离屏渲染器显示这些独立部件，不进行 MuJoCo 物理步进，也不以 MuJoCo 关节运动学重建 Isaac 轨迹。

右侧为参考 MuJoCo `run_grid.run_once` 实际状态回放。先记录实际 qpos/qvel 和有效编译模型，再独立渲染；渲染阶段也不步进物理。
两侧使用相同距离、方位角、俯仰角的摄像机，在固定世界朝向下跟随机器人位置。机器人材质与地面外观来自各自模型，不能把外观差异当作物理差异。

采集：平地、seed 42、初始基座高度 0.74 m、策略周期 0.02 s。Isaac 16 环境，展示第 0 个，物理周期 0.005 s，静/动摩擦 0.8/0.6；MuJoCo 单机器人，物理周期 0.001 s，参考平地滑动摩擦 0.8。
这些既有模型/接触/执行器差异被保留。因此这里只匹配策略和任务指令，不能声称所有仿真条件完全等价。
渲染 25 fps，每段 200 帧 / 8 秒；采集 50 Hz，按最近采样选帧，无轨迹插值。

## 校验与保存

- Isaac 两个模型的 401 个根部 13D 状态（位置、四元数、世界线/角速度）与之前 case 06 的前 8 秒逐项、逐位相同。
- Isaac 各 401 状态、44 部件；MuJoCo 1499 为 401 状态/400 动作，1999 为 131 状态/130 动作。实际命令均为 `[1,0,0]`。
- 1999 的 MuJoCo 原始指标与此前 30 秒计划回放完全一致，除计划时长与生存比例。
- 六个 MP4 均经 ffprobe 验证 25 fps、200 帧、8 秒；人工检查原始姿态网格放置、并排画面及跌倒冻结标记。
- 输入文件哈希在采集前后核对；保存于 `evidence/`。`verification.json` 与 `render_manifest.json` 记录验证与轨迹哈希；`SHA256SUMS.json` 归档交付文件。

原始轨迹、USD 转换网格与有效 MuJoCo 编译模型在被 Git 忽略的本地目录：
`code/day9/g1_balance/checkpoints/flat_baseline/20261002/visual_isaac/` 和 `visual_mujoco/`。
`evidence/raw_files.json` 保存其文件哈希；原始数据并非随 HTML 分发。

采集脚本：`scripts/capture_isaac_visual_poses.py`、`scripts/capture_mujoco_visual_states.py`。
渲染与校验：`scripts/render_simulator_comparison.py`、`scripts/verify_simulator_visual_capture.py`。
视频来自本地训练模型及参考仓库机器人资产；未新训练、commit 或 push。
