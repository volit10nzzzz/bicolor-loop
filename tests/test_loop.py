import tempfile  # 为测试创建隔离的输出目录。
import json  # 检查落盘证据与实际决策一致。
import unittest  # 使用标准库断言。
from pathlib import Path  # 统一临时文件路径。
from bicolor.clients import OfflineClient  # 使用明确标注的模型替身。
from bicolor.graph import Task  # 构造可手算的题目。
from bicolor.loop import solve, run_rounds, validate_proposal  # 调用真实的内外循环。


class ScriptedClient:  # 仅替换模型动作，保留环境与评分。
    def __init__(self, actions):  # 接收预先指定的节点序列。
        self.actions = iter(actions)  # 保留动作顺序。

    def ask(self, role, payload):  # 遵循真实 API 客户端的接口。
        return {"node": next(self.actions)}  # 输出模型应返回的结构化动作。


class LoopTests(unittest.TestCase):  # 检查奖励与跨轮数据依赖。
    def test_interruption_keeps_partial_episode(self):  # 请求中途失败不能丢失已经执行的动作。
        client = ScriptedClient([0])  # 第一条动作成功，第二条触发迭代耗尽异常。
        client.evidence_mode = "offline_simulation"  # 标明测试替身来源。
        with tempfile.TemporaryDirectory() as directory:  # 隔离故障运行。
            root = Path(directory)  # 统一路径。
            with self.assertRaises(StopIteration):  # 原始错误继续传播。
                run_rounds(client, root)  # 在第一道题内中断。
            episode = json.loads((root / "episodes.jsonl").read_text())  # 读取保存的部分回合。
            self.assertEqual(episode["status"], "interrupted")  # 中断不能伪装成终局。
            self.assertEqual(len(episode["steps"]), 1)  # 已执行动作仍可审计。
            self.assertTrue((root / "schedule-0.json").exists())  # 调度条件也保留。

    def test_solver_rewards_and_terminal(self):  # 三次翻转应精确复现讨论中的示例。
        task = Task("manual", "bridge", 2, ((0, 1), (0, 2), (0, 3), (3, 4), (3, 5)), (0, 1, 1, 0, 1, 1), 3)  # 使用六节点题。
        episode = solve(task, ScriptedClient([3, 4, 5]), "r0-q0-e0")  # 执行真实环境。
        self.assertEqual(episode["status"], "solved")  # 满分时立即结束。
        self.assertEqual([step["delta_edges"] for step in episode["steps"]], [-1, 1, 1])  # 核对整数增量。
        self.assertAlmostEqual(sum(step["delta_reward"] for step in episode["steps"]), 0.2)  # 增量仍满足望远镜求和。
        self.assertEqual([step["completion_bonus"] for step in episode["steps"]], [0, 0, 1])  # 完成加成只出现一次。
        self.assertAlmostEqual(sum(step["reward"] for step in episode["steps"]), 1.2)  # 总奖励包括完成加成。

    def test_second_round_uses_first_round_evidence(self):  # 防止外循环只重复随机出题而忽略轨迹。
        with tempfile.TemporaryDirectory() as directory:  # 隔离实验输出。
            records = run_rounds(OfflineClient(), Path(directory), seed=42, rounds=2, rollouts_per_task=2, initial_size=2)  # 跑两轮六条轨迹。
            decision = records[0]["next_proposal"]  # 取第一轮生成的出题决策。
            saved = json.loads((Path(directory) / "rounds.jsonl").read_text().splitlines()[0])  # 读取持久化记录。
            self.assertEqual(saved["next_proposal"], decision)  # 磁盘记录不能丢失下一轮提案。
            ids = {episode["episode_id"] for episode in records[0]["episodes"]}  # 第一轮可引用的证据范围。
            self.assertTrue(decision["evidence_episode_ids"])  # Challenger 必须引用实际轨迹。
            self.assertTrue(set(decision["evidence_episode_ids"]) <= ids)  # 禁止引用不存在的证据。
            self.assertEqual(records[1]["source_round"], 0)  # 第二轮明确指向上一轮。
            self.assertEqual(records[1]["proposal"], decision)  # 第二轮实际消费该提案。
            self.assertEqual(records[1]["tasks"][0]["family"], decision["family"])  # 提案控制目标题型。
            self.assertEqual(len(records[1]["episodes"]), 6)  # 三题各两条轨迹。
            self.assertEqual(records[1]["evidence_mode"], "offline_simulation")  # 模拟结果不能被标成真实 API。
            self.assertIsNone(records[1]["next_proposal"])  # 最后一轮不多付一次无用出题调用。
            self.assertIsNone(records[0]["challenger_reward"])  # 种子轮没有模型出题动作。
            reward = records[1]["challenger_reward"]  # 上一轮出题动作在本轮结算。
            self.assertEqual(reward["reward"], 1.0)  # 四条目标题轨迹全部失败。
            self.assertEqual(reward["decision_id"], "decision-0")  # 延迟奖励归属实际提案。
            self.assertEqual(reward["source_round"], 0)  # 出题观察来自首轮。
            self.assertEqual(reward["evaluation_round"], 1)  # 评价来自下一轮。

    def test_fabricated_evidence_is_rejected(self):  # 模型输出必须引用已存在的上一轮轨迹。
        proposal = {"family": "chain", "size": 2, "hypothesis": "test", "prediction": "solved", "evidence_episode_ids": ["invented"]}  # 构造虚假证据引用。
        with self.assertRaises(ValueError):  # 错误提案明确失败。
            validate_proposal(proposal, [{"episode_id": "real"}])  # 使用真实校验函数。


if __name__ == "__main__":  # 支持单文件执行。
    unittest.main()  # 执行当前测试。
