import json  # 使用已公开的首步请求与响应。
from pathlib import Path  # 在不同工作目录下定位案例。
from unittest import TestCase  # 使用项目已有测试框架。
from unittest.mock import Mock  # 检查求解器真正发出的续轮请求。
from bicolor.case_replay import finish_probe  # 调用真实恢复入口。
from bicolor.graph import generate_task  # 按正式案例种子重建题面。


class ResumeTests(TestCase):  # 避免续跑启动时重复调用模型或消耗不同状态的缓存回答。
    def test_first_step_reused_only_on_matching_observation(self):  # 用一个现有的真实交接覆盖正反两种恢复场景。
        handoff = json.loads((Path(__file__).resolve().parents[1] / "examples/feedback-handoff.json").read_text())  # 使用已完成的真实首步。
        proposal, prefix = handoff["calls"]  # 分别取得第二次出题和接题记录。
        decision = {"output": proposal["output"], "decision_id": proposal["input"]["decision_id"], "planned_round": proposal["input"]["planned_round"], "observation_round": proposal["input"]["round"]}  # 保留结算所需归属。
        task = generate_task(decision["output"]["family"], 4, 242, prefix["input"]["task"]["task_id"])  # 得到真正的第二道目标。
        client = Mock(models=handoff["models"])  # 付费边界完全本地化。
        seen = []  # 记录边界外首次可见的动作数，避免引用会继续增长的历史列表。
        def repeat_node(role, payload):  # 重复翻同一点确保走完整个预算。
            self.assertEqual(role, "solver")  # 恢复后只有解题方需要新回答。
            self.assertEqual(payload["history"][0]["raw_action"], prefix["output"])  # 已执行动作进入每次后续观察。
            seen.append((len(payload["history"]), payload["state"]["remaining_steps"]))  # 明确观察时刻。
            return prefix["output"]  # 固定后续回答，不访问远端。
        client.ask.side_effect = repeat_node  # 只替换模型输出。
        episode, _ = finish_probe(client, task, prefix, decision, [])  # 原动作由真实环境重新执行。
        self.assertEqual(episode["status"], "budget_exhausted")  # 重复翻转耗尽五步。
        self.assertEqual(seen, [(1, 4), (2, 3), (3, 2), (4, 1)])  # 只产生四个后续请求。
        prefix["input"]["state"]["remaining_steps"] -= 1  # 保留题号却改写实际交接状态。
        client.ask.reset_mock()  # 单独核查失败路径的开销。
        with self.assertRaisesRegex(ValueError, "recorded observation mismatch"):  # 错误首步应在任何远端请求之前失败。
            finish_probe(client, task, prefix, decision, [])  # 同一个恢复入口。
        client.ask.assert_not_called()  # 无效缓存不得以重新采样替代。
