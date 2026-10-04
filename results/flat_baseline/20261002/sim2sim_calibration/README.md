# 第一轮 Sim2Sim 校准：接口核验与两个独立变量

先保持 model_1499 为实验对照，model_1999 仍未通过迁移验收。没有重新训练，也没有更改参考仓库、默认运行配置、监测或 supervisor。此轮先验证数字接口，再隔离物理参数差异。

## 实际接口核验

Isaac 每个模型采集 4 秒、200 个真实状态，16 环境中的第 0 个；任务指令 `[1,0,0]`，seed 42。
用 Isaac 的根部四元数、COM 世界线/角速度、关节位置/速度及上一动作，通过参考 MuJoCo `build_policy_observation` 重建输入。
另用导出的 NPZ actor 对实际 Isaac 输入推理，并检查 Isaac 处理后的关节目标是否等于 `default_joint_pos + 0.5 * action`。

两个模型均通过：观测最大绝对误差 `2.3841858e-7`，动作最大绝对误差 `1.1920929e-6`，关节目标误差 0。
关节顺序、默认位置、零默认速度、上一动作和命令字段一致。两次 Isaac 根部 13D 状态与既有 case 06 前 200 样本逐位相同。
这验证的是 **同一状态下的数字接口**；不证明关节轴、碰撞几何、隐式/显式执行器与接触动力学等价。

运行时 37 个关节的 Kp、Kd 均一致。附加关节惯量只有 23/37 一致；力矩上限 0/37 一致，详见 [关节表](joint_contract.csv)。

## 实验 A：只降低四个踝关节力矩上限

model_1499，前向 1 m/s，横向 0，固定转向角速度 0 / -0.2 / +0.2 rad/s，平地摩擦 0.8，seed 42，计划 30 秒。
对照保留 MuJoCo 的 50 Nm 上限；变体只改四个踝关节 `jnt_actfrcrange` 为 ±20 Nm，与 Isaac 踝关节上限一致。
PD、armature、模型资产、策略、命令与步长均保持对照配置。

| 转向指令 rad/s | 50 Nm 对照 | 20 Nm 变体 |
|---|---|---|
| 0 | 完成 30 秒 | 5.54 秒跌倒 |
| -0.2 | 完成 30 秒 | 3.52 秒跌倒 |
| +0.2 | 完成 30 秒，yaw RMSE 0.4535 | 完成 30 秒，yaw RMSE 0.5008 |

该单项改动不能修复迁移；此结果不说明 Isaac 20 Nm 配置本身不合理，因为其余物理系统并未等价。
对照中右踝 pitch 的实际关节驱动力矩超过 20 Nm 的物理步占比为 17.70% / 25.19% / 14.39%；左踝仅 0.41% / 0.51% / 0.36%。这个左右不对称值得继续检查，但原因未确定。
跌倒组只含终止前数据，其 RMSE 不应作为完整 30 秒分数与存活组直接排序。
详见 [实验 A 数据](ankle_comparison.csv)。

## 实验 B：只修正十四个手指关节附加惯量

发现参考 `configure_policy_armature` 通过 `_{index}_joint` 数字后缀识别手指，但它收到的 Isaac 元数据关节名是 `left_five_joint`、`left_zero_joint` 等英文数字名。
因此十四个手指被归为非手指，将其 armature 设置为 0.01；Isaac 实际运行值约 0.001。
本实验使用独立 wrapper，只将这十四项恢复为 0.001，仍保留踝关节 50 Nm 和所有其他配置，没有把 A 与 B 组合。

| 转向指令 rad/s | 对照 yaw RMSE | 修正后 yaw RMSE | 对照 / 修正后前向 RMSE m/s |
|---|---|---|---|
| 0 | 0.4611 | 0.4241 | 0.1215 / 0.1196 |
| -0.2 | 0.5212 | 0.4195 | 0.2360 / 0.1247 |
| +0.2 | 0.4535 | 0.4128 | 0.1312 / 0.1398 |

全部完成 30 秒。三个条件的 yaw RMSE 降低，但仍很大；正转向条件的前向 RMSE 略有增加。不能声称转向已正确，也不能把此结果推广到 model_1999 或其他 seed。
该命名问题值得修复，但本轮只证明其影响，不自动采用变体为默认部署。详见 [实验 B 数据](finger_comparison.csv)。

## 证据与后续

独立核验：源文件哈希未变；三组原始 MuJoCo 对照指标与此前 30 秒 case 06–08 完全一致。
逐项比较编译模型数组：实验 A 仅四个关节的 `jnt_actfrcrange` 改变；实验 B 仅十四个 DOF 的 `dof_armature` 改变。
数值接口与轨迹再次独立核验，见 [验证结果](verification.json)。误差来自参考原始全时段指标，包含启动阶段；yaw 为 body angular z 跟踪误差，不是 Euler 航向变化率。

原始数据在被 Git 忽略的 `code/day9/g1_balance/checkpoints/flat_baseline/20261002/interface_live/`、`ankle_limit_probe/`、`finger_armature_probe/`。
公共目录 `evidence/` 保存输入哈希、运行配置、版本、原始结果；`SHA256SUMS.json` 保存归档哈希。
脚本：`audit_live_isaac_interface.py`、`probe_mujoco_ankle_limits.py`、`probe_mujoco_finger_armature.py`、`analyze_sim2sim_calibration.py`。

下一项优先检查 USD/MJCF 的关节轴、零位、基座/COM 与左右脚碰撞几何，再对隐式执行器和显式 PD 做匹配状态的响应比较。继续保持模型固定和单变量比较；未经补充评估不升级默认基线，不开启新 PPO、崎岖地形训练或 fallback 切换。
