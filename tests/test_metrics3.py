import unittest  # 以手算期望核验统计窗口和分母。
from dataclasses import asdict  # 复用调度器落盘的任务格式。
from bicolor.graph import audit, generate_task  # 用真实任务与证书区分题型和结构难度。
from bicolor.metrics import measure  # 调用运行和回放共享的评价入口。


def scheduled(task_id, group, decision="d", origin=None):  # 构造可审计的调度行。
    family = "chain" if group == "control" else "bridge"  # 历史题故意与目标同题型。
    task = generate_task(family, 2, 42, task_id)  # 同种子使结构可复算。
    return dict(asdict(task), certificate=audit(task), group=group, decision_id=decision, origin_task_id=origin or task_id)  # 附原题标识和样本来源。


def rollout(task, attempt, status, undos=0):  # 构造有明确手算结果的一次作答。
    item = {key: task[key] for key in ("group", "family", "size", "decision_id", "origin_task_id", "task_id")}  # 继承调度归属。
    item.update(episode_id=f"{task['task_id']}-e{attempt}", status=status, final_score=1.0 if status == "solved" else 0.6)  # 保存事实终局和得分。
    if status != "interrupted":  # 运行中断的部分记录没有最终撤销统计。
        item["undo_after_drop"] = undos  # 一条轨迹可能撤销多次。
    return item  # 返回独立样本。


class MetricTests(unittest.TestCase):  # 以固定分组逐项核对奖励和监控。
    def setUp(self):  # 每个测试独立建立两目标一对照。
        self.plan = [scheduled("a", "target"), scheduled("b", "target"), scheduled("c", "control")]  # 每道计划两次作答。
        self.episodes = [rollout(task, i, "budget_exhausted") for task in self.plan for i in range(2)]  # 起始条件全部耗尽且无人撤销。
        self.decision = {"decision_id": "d", "observation_round": 0, "planned_round": 2}  # 预取中观察与评价轮次相距两轮。
        self.proposal = {"family": "bridge", "size": 2, "prediction": "undo_after_drop"}  # 行为预测用于比较命中率。

    def metrics(self, previous=()):  # 调用正式接口并提供固定模型版本。
        return measure(self.proposal, self.episodes, self.decision, 2, self.plan, 2, "solver-v1", previous)  # 保留调度计划，便于发现缺样。

    def test_challenge_and_event_separately(self):  # 全失败与撤销行为是独立测量。
        result = self.metrics()  # 获取全失败、零撤销的结果。
        evaluation = result["challenger_reward"]  # 评价属于生成本轮题的决策。
        self.assertEqual((evaluation["decision_id"], evaluation["source_round"], evaluation["planned_round"], evaluation["evaluation_round"]), ("d", 0, 2, 2))  # 来源不由轮次相减推算。
        self.assertEqual((evaluation["reward"], evaluation["challenge_reward"], evaluation["diagnostic_quality"]), (1, 1, 0))  # 两项分量各自有明确含义。
        for index in (0, 3, 5):  # 目标四中二、对照二中一。
            self.episodes[index]["undo_after_drop"] = 3  # 轨迹中多次撤销只算一次事件。
        result = self.metrics()  # 重算相同命中率的对照。
        self.assertEqual([result["prediction_results"][group]["hits"] for group in ("target", "control")], [2, 1])  # 命中数按回合统计。
        self.assertEqual(result["challenger_reward"]["diagnostic_quality"], 0)  # 两组率相等。
        self.episodes[4]["undo_after_drop"] = 1  # 对照全命中后产生负区分度。
        self.assertEqual(self.metrics()["challenger_reward"]["diagnostic_quality"], -0.5)  # 质量允许负值。

    def test_missing_invalid_and_interrupted_samples(self):  # 检查缺失、不合法和接口故障分别如何落账。
        self.episodes[0] = rollout(self.plan[0], 0, "solved")  # 有效目标一成一败。
        self.episodes[2] = rollout(self.plan[1], 0, "interrupted")  # 异常保留部分轨迹。
        self.episodes.pop(3)  # 另一条已计划但未开始。
        self.episodes[-1]["status"] = "invalid_action"  # 对照含一个无效动作。
        result = self.metrics()  # 模拟中途失败后的监控快照。
        row = next(item for item in result["monitor"] if item["group"] == "target")  # 查询目标层的分母。
        self.assertEqual((row["planned"], row["completed"], row["valid"], row["interrupted"], row["success_rate"]), (4, 2, 2, 1, 0.5))  # 接口故障不进入成功率或终局。
        row = next(item for item in result["monitor"] if item["group"] == "control")  # 查询对照层。
        self.assertEqual((row["planned"], row["completed"], row["valid"], row["invalid_action"]), (2, 2, 1, 1))  # 无效动作有终局但无有效评价。
        self.assertIsNone(result["challenger_reward"]["reward"])  # 未完成目标计划不可部分结算。
        self.assertIsNone(result["challenger_reward"]["diagnostic_quality"])  # 行为比较需要双方完整。
        self.episodes = [rollout(task, i, "solved") for task in self.plan for i in range(2)]  # 恢复完整批次。
        self.episodes.pop()  # 只有对照缺失。
        result = self.metrics()["challenger_reward"]  # 评价每个分量的依赖。
        self.assertEqual(result["challenge_reward"], 0)  # 目标完整时挑战分有效。
        self.assertIsNone(result["diagnostic_quality"])  # 对照缺失时质量未知。

    def test_history_and_decision_attribution(self):  # 同题型历史题及其他决策都不能影响当前分母。
        original = scheduled("orig", "target", None)  # 原题属于种子轮。
        initial = [rollout(original, i, "solved") for i in range(2)]  # 首次两条全部成功。
        previous = [{"round": 0, "proposal": self.proposal, "tasks": [original], "episodes": initial}]  # 最早可见记录。
        historical = scheduled("again", "historical_retest", None, "orig")  # 给重测独立作答编号。
        self.plan.append(historical)  # 重测也进入监控计划。
        self.episodes.extend(rollout(historical, i, "budget_exhausted", 1) for i in range(2))  # 两条全败且都有撤销。
        self.episodes.append(rollout(scheduled("foreign", "target", "other"), 0, "solved", 1))  # 模拟同题型另一决策的轨迹混入。
        result = self.metrics(previous)  # 评价仅归当前动作。
        self.assertEqual((result["challenger_reward"]["target_episodes"], result["challenger_reward"]["reward"]), (4, 1))  # 历史与其他决策不污染奖励。
        self.assertEqual(result["prediction_results"]["target"]["hits"], 0)  # 历史撤销也不污染预测。
        row = result["retests"][0]  # 多条重测合并回原题。
        self.assertEqual((row["origin_task_id"], row["original"]["episodes"], row["retest"]["episodes"]), ("orig", 2, 2))  # 保留首次与重测次数。
        self.assertEqual((row["success_rate_change"], row["undo_rate_change"]), (-1, 1))  # 报告前后的有向变化。
        self.assertFalse(result["coverage"]["new_target_structure"])  # 同题型同规模仍是已见结构。

    def test_terminal_prediction_and_seed(self):  # 终局预测只报告命中数。
        self.proposal["prediction"] = "budget_exhausted"  # 预注册终局事件。
        result = self.metrics()  # 计算原始目标和对照命中。
        self.assertEqual(result["prediction_results"]["target"]["hits"], 4)  # 命中可观测。
        self.assertEqual(result["challenger_reward"]["quality_status"], "not_applicable")  # 避免重复给失败率打分。
        self.assertIsNone(result["challenger_reward"]["diagnostic_quality"])  # 不适用保持空值。
        self.decision = None  # 种子没有出题模型动作。
        self.assertIsNone(self.metrics()["challenger_reward"])  # 种子观察永不归功出题方。
