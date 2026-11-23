# 双角色奖励与训练接入契约（版本 2）

本文保留历史 v2 数据契约。当前入口的分项奖励、同步与预取调度见 [v3 课程与反馈](curriculum.md)。

## Solver

由确定性图环境评分，模型只输出动作。`score = satisfied_edges / edge_count`。

`delta_reward = next_score - score`

`completion_bonus = B if first_success else 0`

`reward = delta_reward + completion_bonus`

B 由 config.toml 的 run.completion_bonus 指定，默认1，必须有限且非负。成功立即结束，因此最多一次完成奖励。本版目标不折扣（gamma=1），累计奖励等于最终分数减初始分数，加上成功加成。没有额外重复计入最终分数。

无效节点：当步增量与加成为0，回合以 invalid_action 终止，训练导出跳过整条回合。HTTP/解析/预算故障：标记 interrupted 并停止运行，保留已执行步骤，不把故障作为解题失败。

## Challenger

当前目标是寻找同一规模下 Solver 的可验证弱点，不是直接优化学习进步。输入上一轮统计、完整轨迹、前一提案及 fixed_size，输出 family、size、hypothesis、prediction、evidence_episode_ids。

size 必须等于本次 initial_size；题型限 bridge/chain。程序核验题目可解且预算足够，生成两道目标题与一道对照题。

`reward = 目标题中 budget_exhausted 回合数 / 目标题回合数`

只有目标题所有回合均为 solved 或 budget_exhausted 才计分；包含 invalid_action 或 interrupted，或没有目标题时，status=inconclusive、reward=null。对照组不参与奖励，行为预测命中率独立报告。不对纯格式失败给奖励，也不通过增加规模给奖励。

出题动作来自第 k 轮观察，奖励来自第 k+1 轮求解。记录包含 decision_id、source_round、evaluation_round、episode_ids。种子轮 challenger_reward=null，不能算作模型出题成果。两轮示例恰好产生一个可评价的 Challenger 决策。

目前两个同规模题型构成有限动作空间；新节点编号不是新结构。coverage.structure_key=family:size，new_target_structure 标记是否首次作为目标题。重复同构题是重复测量，固定2:1比例，未实现多样性奖励、自动分层概率采样或防止长期重复的课程调度。失败率奖励可能收敛到较难题型，不足以证明持续高质量课程。

## 可复算与可复现

`python3 -m bicolor.replay --run runs/demo`

回放读取当时动作与提案，不调用模型；用相同种子重建任务，通过真实环境重新计算状态、奖励、统计与下一轮归属，再逐字段对比 rounds、episodes 和 decision 文件。缺失字段、错误奖励、观察不一致或未完成运行均不能通过。只支持 schema_version=2；旧示例保留为历史格式，不混用。

回放只能证明内部一致性，不能证明原始文件未经联合篡改，也不能证明服务商身份或模型能力。真实 API 响应具有随机性，固定种子只固定任务，无法保证重复调用得到相同答案。

## 训练出口

`python3 -m bicolor.train --run runs/demo --role solver --output runs/demo/solver-transitions.jsonl`

`python3 -m bicolor.train --run runs/demo --role challenger --output runs/demo/challenger-transitions.jsonl`

导出前先完整回放核验。Solver 样本含 role、task、state、history、action、delta_reward、completion_bonus、reward、next_state 和终止标记。预算已纳入状态，因此预算终止是 terminated，而非 truncated。

Challenger 样本含原始 observation、提案 action、延迟 reward、评价轨迹引用。每个提案按独立的一步出题任务处理，terminated=true，不声称建模多轮出题策略的长期回报。inconclusive 提案不导出，报告跳过数量。

以上是环境级数据契约，不是 PPO batch。后续训练器需要本地模型采样、token 对齐、动作范围、行为概率，以及所选算法需要的 value/优势估计和更新实现。两角色的训练节奏与策略版本也需要训练器明确管理。本版不会启动训练，未验证任何训练适配器。

## API 验证状态

v2 验收时使用本机 HTTP 替身检查客户端和双角色循环。当前公开版本附带可离线复算的双轮 API 案例，并保留独立配置服务和模型的入口，见 [运行与验证](../README.md)。
