## 项目状态 / Status

> 当前状态：**暂停维护（Archived）**  
> 本仓库是基于开源引擎 `gamerpuppy/sts_lightspeed` 的个人研究与重构分支，主要用于强化学习（RL）实验，不作为商业产品或对外服务。

---

# 1. 项目简介（Overview）

本仓库是在原始 C++ 引擎 [gamerpuppy/sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) 的基础上，增加了：

- 面向强化学习的 Python 绑定与 Gymnasium 环境封装  
- 观测向量、奖励函数、路径规划等一整套 AI 训练管线  
- 针对训练稳定性的大量防御性改造与 Bug 修复  

可以理解为：在保留原有高性能 Slay the Spire 模拟器的前提下，叠加了一整层 RL 实验平台和工程化加固，是一次**比较彻底的二次开发 / 重构**。

---

# 2. 上游项目与开源协议（Upstream & License）

- 上游项目：[`gamerpuppy/sts_lightspeed`](https://github.com/gamerpuppy/sts_lightspeed)  
- 上游作者：`gamerpuppy`  
- 上游协议：MIT License  
- 本仓库根目录的 [LICENSE.md](./LICENSE.md) 完整保留了原作者的 MIT 许可证声明。

**MIT 协议的关键点（非法律意见）：**

- 允许任何人免费使用、复制、修改、合并、发布、分发、再授权、出售软件副本  
- 前提是：在软件及其重要部分中保留版权声明和许可声明  
- 作者不对使用结果承担任何形式的责任

本仓库已经：

- 保留原始 MIT 许可证内容及版权声明  
- 明确标注上游仓库地址与作者信息  

从开源协议的角度，以上做法通常被视为**符合 MIT 许可要求**。  
但需要强调：**我并非法律专业人士，以下内容不构成法律意见**；如果未来涉及商业化或对外分发，请务必自行咨询专业律师。

此外，本项目只是对游戏《Slay the Spire》的第三方模拟与研究工具，不代表或隶属于官方开发商 MegaCrit。

---

# 3. 在原项目基础上做了哪些工作？（What Has Been Added/Changed）

本仓库的改动大致可以分为四层：

## 3.1 C++ 引擎层（Engine Level）

- 维护并继续使用原有高性能 C++17 引擎，保持：
  - 目标：接近 100% RNG 一致性  
  - 支持 Ironclad、遗物、事件、地图、三幕流程等核心机制  
- 在此基础上进行了一些有针对性的修改和修复：
  - **地图坐标与 Boss 楼层问题修复**  
    - 针对 Act 1 Boss（Y=15）等场景，修正了地图跳转过程中可能触发的越界访问和 `CRITICAL ERROR` 级崩溃。  
    - 对跨幕时 `Y = -1` 这一特殊含义进行了逻辑梳理和安全加固（详细见 [ENGINE_HANDOVER_GUIDE.md](./ENGINE_HANDOVER_GUIDE.md)）。
  - **战斗逻辑 Bug 修复**  
    - 修复了如 `Dual Wield` 等特定卡牌在复制和手牌重排时导致的不一致行为。  
  - **暴露更多控制接口**  
    - 通过 C++ 与 Python 绑定，额外暴露了 `play_card`、`end_turn`、`choose_map_node`、`choose_event_option`、`choose_campfire_option`、`choose_treasure_open`、`regain_control` 等接口，使 Python 侧可以精细控制游戏流程。

整体上，C++ 部分仍然是**基于原有结构的小步扩展 + 定向修复**，没有推倒重写，但已经构成一次比较系统的“工程化升级”。

## 3.2 Python 绑定与环境层（Bindings & Env）

新增并改造了完整的 Python 接入层，用于强化学习：

- 使用 **pybind11** 将 C++ 引擎封装为 Python 模块（`slaythespire`），供上层环境调用。  
- 实现自定义 Gymnasium 环境：
  - [sts_env.py](./sts_env.py)：`StsEnv`  
  - 提供标准的 `reset` / `step` / `action_space` / `observation_space` 接口  
  - 增加动作掩码（Action Mask）和多种防御性检查，避免非法操作导致引擎崩溃  
- 为了便于调试和训练，还补充了：
  - [train_check.py](./train_check.py)：快速检查环境是否可以正常 reset / step  
  - [test_env.py](./test_env.py)：长时间压力测试 `regain control` 相关逻辑  
  - [final_gate_test.py](./final_gate_test.py)：验证跨幕（Act 2 → Act 3）与胜利结算的逻辑连通性

## 3.3 观测与奖励设计（Observation & Reward）

- [observation.py](./observation.py)  
  - 设计并实现了一个约 72 维的观测向量，将玩家状态、怪物、手牌、全局信息打平成固定维度的数值输入，适配深度强化学习模型。  
- [reward.py](./reward.py)  
  - 为 Ironclad 设计了一个“理性（Rational）”奖励函数：  
    - HP 视为资源：小损血容忍，大量失血惩罚  
    - 强化收益：力量增长、牌组瘦身、金币累积给予正向奖励  
    - 击杀怪物、推进楼层给予进度型奖励  
  - 支持输出奖励拆解信息，便于调参和分析。

这些部分基本上属于**强化学习专用的新增逻辑**，在原仓库中不存在。

## 3.4 路径规划与策略评估（Path Planning & Evaluation）

- [map_evaluator.py](./map_evaluator.py)  
  - 实现了一个 `MapEvaluator`，用于对整张地图路径进行打分和选择：  
    - 考虑当前 HP、金币等因素动态调整偏好  
    - 在 HP 较低时自动规避精英路线，在金币较多时优先商店  
    - 使用 DFS + 记忆化优化实现合理的搜索效率  
- 配合环境与奖励，可用于构建“理性”路线上升的智能体，而不仅仅是随机游走。

---

# 4. 训练脚本与性能（Training & Performance）

## 4.1 训练脚本

- [train_ppo.py](./train_ppo.py)  
  - 基于 `stable-baselines3` 的 PPO 算法  
  - 使用 `SubprocVecEnv` 启动多进程环境（默认 8 个环境）  
  - 带有 TensorBoard 日志、定期 Checkpoint 保存、自动恢复训练等功能  
  - 目标步数配置为 **6000 万步**，适合长时间过夜训练  
- [enjoy_ppo.py](./enjoy_ppo.py)  
  - 用于加载训练好的模型并进行可视化评估  
  - 通过 `StsEnv` 逐步回放策略行为，打印楼层与 HP 变化。

## 4.2 性能参考（仅供参考）

由于强化学习涉及 Python、C++ 与多进程，性能与硬件、系统环境关系较大，以下仅为在开发机上的粗略观测：

- 原始 C++ 引擎：上游作者报告为“16 线程下 5 秒 100 万随机对局”量级  
- 本 RL 分支：  
  - 在启用 8 个并行环境、开启 CUDA 的情况下，环境步数吞吐大致在 **几千步 / 秒** 量级（约 3000 steps/s 级别）  

具体数值会因显卡、CPU、系统设置等发生明显波动，请以实际运行情况为准。

---

# 5. 已知问题与潜在风险（Known Issues & Risks）

## 5.1 引擎层面

- 部分 C++ 断言 / `CRITICAL ERROR` 仍然存在于原始代码路径中：  
  - 在极端或尚未覆盖到的状态组合下，仍有触发进程级崩溃的理论可能。  
  - Python 侧的环境封装已经通过动作掩码和额外判定尽量避免触发这些路径，但不能保证 100% 覆盖。  
- 某些日志中仍可能出现如 `regain control lambda was null` 之类的错误信息：  
  - 目前观察是偶发、但不会阻断训练流程（通过额外防御逻辑和重置机制绕过）。  
  - 根因定位与彻底消除仍需要更深入的引擎级分析。

详细的危险点、状态机分析和降级策略，已经在 [ENGINE_HANDOVER_GUIDE.md](./ENGINE_HANDOVER_GUIDE.md) 中整理。

## 5.2 游戏完整性与策略表现

- 当前重点在 Ironclad + 普通流程，对其他角色和特定事件/遗物的系统性验证仍不充分。  
- 强化学习 Agent 距离“稳定通关整局游戏”还有很大距离，目前更多是**框架与工具层面已经就绪**，策略质量仍处于探索阶段。  

## 5.3 法律与合规风险提示（非法律意见）

- 引擎本身基于 MIT 协议开源，遵守许可证条款（保留版权与许可证）后可以自由修改和分发。  
- 但本项目模拟的是商业游戏《Slay the Spire》的玩法与机制：  
  - 在个人研究与非商业用途范围内，一般风险较低；  
  - 若计划对外发布、商业化或与第三方服务集成，建议事先咨询专业律师，并留意游戏版权方的政策与条款。  

本仓库作者仅将其用于个人技术研究与 AI 训练实验，不对任何第三方使用方式承担责任。

---

# 6. 使用方式（How to Use）

以下为一个粗略的使用指引，假设你已经在本地编译好 C++ 核心并生成 Python 模块（通常在 `build/Release` 目录下）：

1. 确保 Python 环境中已安装：
   - `gymnasium`  
   - `stable-baselines3`  
   - `torch`（如需 GPU，则安装支持 CUDA 的版本）  
2. 确保 `build/Release` 已被加入 `PYTHONPATH` 或由脚本自动追加（例如 `train_ppo.py` 中的逻辑）。  
3. 运行训练脚本：
   ```bash
   python train_ppo.py
   ```
4. 使用 TensorBoard 观察训练过程：
   ```bash
   tensorboard --logdir ./tensorboard_logs
   ```
5. 训练完成后，可通过：
   ```bash
   python enjoy_ppo.py
   ```
   进行若干回合的可视化评估。

> 由于当前仓库状态是“暂停维护”，上述流程更偏向“记录当时是如何跑起来的”，不保证在未来环境或依赖升级后仍可直接使用。

---

# 7. 项目状态与后续方向（Project Status & Future Direction）

- 当前状态：  
  - 引擎 + 环境 + 训练脚本已能支持长时间（数千万步级）PPO 训练；  
  - 已完成基础的鲁棒性加固和关键逻辑梳理；  
  - 由于引擎深层次问题和整体工程复杂度，**目前选择暂时搁置**。  

- 若未来重新启动本项目，优先方向可能包括：  
  - 进一步清理和统一 C++ 状态机与异常处理，使所有错误都能安全回落到“环境重置”而非进程崩溃；  
  - 构建更加模块化、可复用的观察与奖励配置系统；  
  - 引入更强的策略（例如基于 Transformer 的决策模型）以及更系统的评估基准。

---

# 8. 致未来的自己 / 接手的开发者

- 本仓库最强的地方在于：  
  - 你已经拥有一个**可被机器学习直接消费的 Slay the Spire 高性能环境**；  
  - 引擎细节和关键安全边界已经在 [ENGINE_HANDOVER_GUIDE.md](./ENGINE_HANDOVER_GUIDE.md) 中整理清楚。  
- 同时，它也是脆弱的：  
  - 深层 C++ 逻辑仍然存在历史包袱和潜在边界条件；  
  - Python 与 C++ 的交界处仍需要极高的严谨度来避免“自杀式”崩溃。  

如果哪一天你决定让这个 AI 真正“通关游戏”，建议从以下几点入手：

1. 先把所有可能导致进程直接退出的路径彻底梳理并改造成安全的错误返回。  
2. 建立一套自动化的“跑完全程”测试（包含三幕、不同路线和关键事件），保证引擎永远不会卡死或崩溃。  
3. 在一个足够稳定的引擎版本上，再去精细打磨策略和算法，否则训练出来的只是“学会在 Bug 周围跳舞”的智能体。  

当你再次打开这个仓库时，希望这些文档能帮你在最短时间内“接上昨天的梦”。
