# G1研究证据链索引（2026-10-03）

这是证据索引，不是最终申请展示包。各节点保留自己的协议、结果与限制；公开结果在results，原始大轨迹/权重主要在Git忽略的本地checkpoints与参考仓库logs中。没有commit/push，不代表这些成果已经上传。

| 节点／研究问题 | 已保存的依据 | 当前结论／限制 |
|---|---|---|
| 从零能否学会行走 | [训练复现](../results/locomotion_reproduction/README.md)，1499权重、训练日志、评估指标 | 123D/37D、本地从零训练；不是完整复现参考控制品质 |
| 是否应直接追加PPO | [1499/1999对照](../results/flat_baseline/20261002/posttraining/README.md)，双引擎匹配轨迹与权重来源 | Isaac部分改善，Mu失败增加；1999未采用 |
| 监测是否干预动作 | [六组转向配对](../results/locomotion_shadow/turning/README.md)及[物理命令矩阵](../results/locomotion_shadow/physical_commands/README.md) | 开关配对一致；不能据此认为异常检测有效 |
| 为什么持续偏差漏检 | [窗口诊断](../results/locomotion_shadow/yaw_window/README.md)、[检测校准](../results/locomotion_shadow/yaw_calibration/README.md) | 坐标/窗口/阈值分析；自动切换资格未通过 |
| 跨仿真偏差的来源 | [驱动对照](../results/flat_baseline/20261002/matched_actuation/README.md)、[腿部接触](../results/flat_baseline/20261003/leg_contact/README.md)、[接触参数](../results/flat_baseline/20261003/contact_parameters/README.md) | 多项配置/动力学证据；没有确定唯一原因 |
| 高层任务与失联时序能否验证 | [任务演示](../results/hierarchical_demo/20261003/README.md)、[交接处watchdog](../results/hierarchical_demo/20261003/handoff/README.md) | 编排/有效性/时序已测试；保留命令/结束仿真不等于安全停车 |
| 零指令或静态关节姿态是否可停车 | [命令响应](../results/hierarchical_demo/20261003/command_response/README.md)、[姿态保持](../results/hierarchical_demo/20261003/posture_hold/README.md) | 失败对照保留；未当作fallback |
| 两个新技能能否训练 | [首轮](../results/skills_sprint/20261003/first_round/README.md)，独立model999、配置、日志、216案例原始trace | 局部技能出现；全指标通过0，不能等同完全未学会 |
| 是技能问题还是交接问题 | [684案例分离诊断](../results/skills_sprint/20261003/handoff_diagnostic/README.md) | 静止启动/direct/ramp配对；交接影响存活，Mu仍有持续偏差 |
| 覆盖真实1499接手状态能否改善 | [接手训练500](../results/skills_sprint/20261003/teacher_handoff_round/README.md)，seed43状态库、seed44训练、seed42评估、model499与原始trace | Isaac停车31/96、踏步存活96/96；Mu未验收，原指标不改 |
| 固定actor时上层航向反馈有无收益 | [324案例反馈对照](../results/skills_sprint/20261003/heading_feedback/README.md)，增益0/.5/1、配对轨迹与独立重建 | Isaac踏步航向改善，强反馈停车退化，Mu未解决；未采用通用增益 |

## 每个节点应如何进入提交材料

按“问题 → 固定条件 → 唯一改动或明确改动集合 → 对照数据 → 解释 → 下一步”整理。区分训练奖励、程序核验通过与控制性能通过；记录失败实验、采用/不采用的理由。指向对应结果表、运行配置、权重/模型SHA、原始轨迹、验证报告；有视频的节点再附视频，不暗示所有节点都有视频。

## 保存与来源限制

历史CSV有些缺少完整运行环境/模型快照，不能事后补成严格复现证据。今天计算的hash只说明当前文件内容，不代表训练时已捕获同样快照。最新两轮训练权重、配置、原始评估状态/观测/动作、日志与SHA都在本地；大文件被Git忽略，之后打包时应决定随附、链接或保留下载说明。

[机器可读库存](skills_sprint/20261003_evidence_inventory.json)记录18个节点及本轮技能原始资产hash，节点递归计数有重叠，不是独立实验数量。总项目状态以G1_PROGRESS.md最后记录为准。

## State-based multi-skill sequence and watchdog stop/resume (completed 2026-10-04 Sydney)

New evidence: [experiment report](/home/omai/robotics/projects/g1-hierarchical-control/results/skills_sprint/20261003/skill_sequence/README.md).
Frozen walking1499 + independently trained hold499, no new PPO.144prospective low-speed sequence cases:direct48/48,blend43/48,brake+blend37/48. Explicit causal readiness gate and both actor directions tested.96task-heartbeat fault cases:temporary48/48stop/resume,persistent48/48settled12s holding with no unauthorized resumption. These limited conditions do not replace standalone skill acceptance or establish actual perception-dropout/hardware robustness.240case results,226254first-episodeaction rows independently verified-O,144paired prefixes exact,15formal raw tracesSHA preserved. Source snapshots,environment/input/model hashes,events,metrics,comparisonfigure and a labeled19.2s replay video saved. Raw data locally ignored, not GitHub published. Negative blending/braking comparison retained rather than assuming transition complexity helps.

## Held-out sequence replication and bounded demo freeze (2026-10-04 Sydney)

[Retest report](/home/omai/robotics/projects/g1-hierarchical-control/results/skills_sprint/20261004/sequence_retest/README.md): frozen previous policies/gates/direct handoff,324new cases(seeds101/202/303,entry3.22/6.22/9.22s,yaw±.2/0). Normal108/108,temporarydrop108/108,persistent12sactorobservation108/108; independently reconstructed318455actions and216paired prefixes,explicit horizonschecked. Record stopping net/path,holding drift,communication detection vsphysical confirmation andresume tracking latency. Reset seeds vary world root pose only; no perturbation/noise/rough/Mu/hardware claim. `frozen_demo.json` captures a reproducible experimental version; no production promotion or originalskillcriteria relaxation. Fullrawtraces/models/logslocallyignored,notGitHubuploaded.
