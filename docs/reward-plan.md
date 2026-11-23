# 双角色奖励实施计划

此文档记录 v2 奖励实现范围，当前版本与公开案例见 [项目说明](../README.md)。

- [x] Solver：reward=delta_reward+completion_bonus，默认完成加成1，首次成功即终止；无效动作增量与加成均为0；API异常不算任务失败。
- [x] Challenger：规模锁定为 initial_size，仅选择题型；后续目标题有效完成回合的失败比例为奖励。任何无效动作或中断使此次评价 inconclusive，reward=null，避免以格式问题获奖。种子轮没有 Challenger 奖励。
- [x] 建立 decision_id、source_round、evaluation_round、目标轨迹引用；按同构题型聚合覆盖，明确重复编号不是新增覆盖。
- [x] 加入保存轨迹重放与奖励复算命令。篡改或缺失字段必须失败，离线/API/重放来源分开记录。
- [x] 双角色导出环境级训练数据契约，不伪造 token 概率，不实现优化器。
- [x] 手算、延迟归属、固定难度、故障、回放篡改、CLI及HTTP测试；更新示例、README、架构图与验证边界。

v2 采用固定规模，每轮两道目标题和一道对照题。同构题按重复测量记录。该版本使用离线替身验收，API 客户端保留可配置的模型入口。
