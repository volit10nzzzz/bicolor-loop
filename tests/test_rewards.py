import json  # 操作真实运行记录以测试错误证据。
from pathlib import Path  # 管理临时路径。
import subprocess  # 使用正式命令行生成证据。
import sys  # 沿用测试解释器。
import tempfile  # 隔离测试文件。
import unittest  # 使用标准库断言。
from bicolor.loop import evaluate_challenger, validate_proposal, solve  # 检验真实奖励实现。
from bicolor.graph import Task  # 构造可手算的成功题。
from bicolor.replay import verify_run  # 重算双方奖励。
from bicolor.train import export_transitions  # 检查训练角色契约。
from test_loop import ScriptedClient  # 仅控制模型动作。


class RewardTests(unittest.TestCase):  # 检验完成加成和延迟归属的边界。
    def test_configurable_bonus_and_invalid_action(self):  # 配置应真正改变完成奖励。
        task = Task("manual", "bridge", 2, ((0, 1), (0, 2), (0, 3), (3, 4), (3, 5)), (0, 1, 1, 0, 1, 1), 3)  # 与手算例子一致。
        episode = solve(task, ScriptedClient([3, 4, 5]), "e", completion_bonus=2)  # 不依赖模型碰巧成功。
        self.assertAlmostEqual(sum(step["reward"] for step in episode["steps"]), 2.2)  # 首次成功加二分。
        invalid = solve(task, ScriptedClient([99]), "bad")  # 非法节点是任务动作错误。
        self.assertEqual(invalid["status"], "invalid_action")  # 不伪装成正常失败。
        self.assertEqual(invalid["steps"][0]["completion_bonus"], 0)  # 无效动作不获得完成奖励。

    def test_challenger_excludes_control_and_faults(self):  # 失败率只作用于有效目标题。
        proposal = {"family": "bridge"}  # 此函数消费已校验提案。
        episodes = [{"family": "bridge", "episode_id": "a", "status": "solved"}, {"family": "bridge", "episode_id": "b", "status": "budget_exhausted"}, {"family": "chain", "episode_id": "c", "status": "budget_exhausted"}]  # 目标题一成一败。
        self.assertEqual(evaluate_challenger(proposal, episodes, 1)["reward"], 0.5)  # 对照失败不改变分母。
        for status in ("invalid_action", "interrupted"):  # 两类不完整测量都不能获奖。
            episodes[0]["status"] = status  # 注入一个异常回合。
            result = evaluate_challenger(proposal, episodes, 1)  # 使用实际结算函数。
            self.assertIsNone(result["reward"])  # 未知不是零或满分。
            self.assertEqual(result["status"], "inconclusive")  # 明确评价不足。
        self.assertIsNone(evaluate_challenger(proposal, [], 1)["reward"])  # 空样本也不能计分。

    def test_fixed_size(self):  # 不能靠扩大题目规模提高出题奖励。
        proposal = {"family": "bridge", "size": 3, "hypothesis": "same", "prediction": "solved", "evidence_episode_ids": ["a"]}  # 超出本次固定组。
        with self.assertRaisesRegex(ValueError, "fixed"):  # 外部提案被拒绝。
            validate_proposal(proposal, [{"episode_id": "a"}], fixed_size=2)  # 保持难度条件可比较。

    def test_replay_export_and_tampering(self):  # 原始证据通过，损坏或缺少字段必须失败。
        with tempfile.TemporaryDirectory() as directory:  # 不覆盖演示数据。
            root, run = Path(directory), Path(directory) / "run"  # 指定新运行。
            subprocess.run([sys.executable, "-m", "bicolor", "--output", str(run)], check=True, capture_output=True)  # 跑真实两轮。
            self.assertEqual(verify_run(run)["status"], "passed")  # 回放应逐字段一致。
            export_transitions(run, root / "challenger.jsonl", "challenger")  # 导出延迟奖励样本。
            sample = json.loads((root / "challenger.jsonl").read_text())  # 两轮恰好一次出题动作。
            self.assertEqual(sample["reward"], 1.0)  # 继承真实目标题失败比例。
            self.assertEqual(sample["observation"]["round"], 0)  # 不能把下一轮观察泄露给出题动作。
            manifest_path = run / "manifest.json"  # 检验来源不能只靠一个标签伪造。
            saved_manifest = manifest_path.read_text()  # 保留真实配置。
            manifest = json.loads(saved_manifest)  # 构造来源矛盾。
            manifest["evidence_mode"] = "api"  # 将离线记录误标为模型数据。
            manifest_path.write_text(json.dumps(manifest))  # 保存错误样本。
            with self.assertRaisesRegex(ValueError, "source"):  # 必须拒绝来源混淆。
                verify_run(run)  # 核对配置、轮次和清单。
            manifest_path.write_text(saved_manifest)  # 恢复后继续奖励篡改测试。
            path = run / "rounds.jsonl"  # 在保存证据上注入错误。
            original = path.read_text()  # 保存原始记录供下一场景复用。
            records = [json.loads(line) for line in original.splitlines()]  # 读取可修改副本。
            records[1]["challenger_reward"]["reward"] = 0.25  # 伪造出题奖励。
            path.write_text("\n".join(json.dumps(item) for item in records))  # 写入坏样本。
            with self.assertRaises(ValueError):  # 奖励复算必须拒绝。
                verify_run(run)  # 不依赖模型评审。
            records = [json.loads(line) for line in original.splitlines()]  # 恢复并构造缺失字段场景。
            del records[0]["episodes"][0]["steps"][0]["delta_reward"]  # 缺失关键奖励分解。
            path.write_text("\n".join(json.dumps(item) for item in records))  # 保存不完整记录。
            with self.assertRaises(ValueError):  # 不完整证据不能误判通过。
                verify_run(run)  # 重放生成完整字段后比较。
