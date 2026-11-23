import io  # 用内存字节流提供标准 HTTP 响应。
import json  # 检查真实序列化后的模型请求与日志。
import tempfile  # 隔离各次运行和导出文件。
import unittest  # 使用标准库运行离线回归。
from concurrent.futures import ThreadPoolExecutor  # 在真实线程之间交错共享客户端请求。
from pathlib import Path  # 管理调用日志和训练样本。
from threading import Event  # 明确控制第一条请求的返回时点。
from unittest.mock import patch  # 隔离远端网络边界。
from bicolor.clients import ChatClient, OfflineClient  # 直接测共享客户端及冻结的规则策略。
from bicolor.train import export_transitions  # 检查双角色训练数据。

class ClientContractTests(unittest.TestCase):  # 检验请求记账、反馈与训练契约。
    def test_reverse_completion_and_shared_budget(self):  # 出题先发后回，求解失败仍消耗最后一次额度。
        entered, release = Event(), Event()  # 在不靠等待时长猜测的情况下控制并发。
        def respond(request, timeout):  # 仅替换 HTTP 响应侧。
            if json.loads(request.data)["model"] == "c":  # 阻塞先到的出题调用。
                entered.set()  # 表明第一条请求已进入发送边界。
                if not release.wait(5):  # 为失败断言提供有限退出条件。
                    raise TimeoutError("test release missing")  # 避免测试线程死锁。
                return io.BytesIO(b'{"choices":[{"message":{"content":"{}"}}]}')  # 返回一个正常 JSON 对象。
            raise OSError("offline transport failure")  # 已发出的求解请求在网络边界失败。
        with tempfile.TemporaryDirectory() as directory, patch("bicolor.clients.urlopen", side_effect=respond) as transport:  # 全程不连接模型服务。
            log = Path(directory) / "calls.jsonl"  # 独立记录两条调用。
            client = ChatClient("https://example.invalid", "local", "s", "c", log, max_calls=2)  # 两角色共享两次总额度。
            with ThreadPoolExecutor(max_workers=2) as pool:  # 仅控制返回，不串行化两条请求。
                first = pool.submit(client.ask, "challenger", {"decision_id": "d", "feedback": {}})  # 第一次请求占用一个额度。
                try:  # 即使中途断言失败也释放第一条请求。
                    self.assertTrue(entered.wait(5))  # 第一条到达后再发第二条。
                    with self.assertRaises(OSError):  # 保留底层传输失败。
                        client.ask("solver", {"episode_id": "e", "history": []})  # 第二条比第一条先结束。
                    with self.assertRaisesRegex(RuntimeError, "budget"):  # 超额调用在发送边界之前拒绝。
                        client.ask("solver", {"episode_id": "e", "history": []})  # 重复提交也不能突破上限。
                finally:  # 任何断言路径都允许线程退出。
                    release.set()  # 允许第一条请求写完日志。
                self.assertEqual(first.result(), {})  # 原请求仍返回自己的成功结果。
            calls = [json.loads(line) for line in log.read_text().splitlines()]  # 按落盘顺序读取。
            self.assertEqual([(c["call"], c["request_id"], c["status"]) for c in calls], [(2, "request-2", "error"), (1, "request-1", "ok")])  # 失败与乱序均保留真实发送编号。
            self.assertEqual((client.calls, transport.call_count), (2, 2))  # 预算耗尽没有制造第三个实际开销。

    def test_feedback_changes_next_behavior_probe(self):  # 检验确定性替身确实读取已结算的行为区分度。
        episode = {"episode_id": "r1-q0-e0", "undo_after_drop": 1, "status": "budget_exhausted"}  # 当前轨迹上可观察到撤销。
        observation = {"fixed_size": 2, "episodes": [episode]}  # 旧输入还没有专门反馈字段。
        client = OfflineClient()  # 直接使用将供命令行调用的策略。
        legacy = client.ask("challenger", observation)  # 旧版仍会在 bridge 上复测撤销。
        self.assertEqual((legacy["family"], legacy["prediction"]), ("bridge", "undo_after_drop"))  # 固定旧版契约。
        feedback = {"decision_id": "decision-0", "proposal": legacy, "challenger_reward": {"diagnostic_quality": 0}}  # 此前 bridge 对照实验对撤销没有区分度。
        alternative = client.ask("challenger", dict(observation, feedback=feedback))  # 只改变已结算反馈。
        self.assertEqual((alternative["family"], alternative["prediction"], alternative["evidence_episode_ids"]), ("chain", "undo_after_drop", ["r1-q0-e0"]))  # 在相反结构上检验相同事件并引用当前可见轨迹。

    def test_training_exports_preserve_versions_and_pre_action_context(self):  # 旧证据可继续导出，新训练输入只含动作前可见内容。
        import subprocess  # 使用正式运行入口生成预取与重测数据。
        import sys  # 调用当前测试解释器。
        project = Path(__file__).resolve().parents[1]  # 定位随仓库保存的旧证据。
        with tempfile.TemporaryDirectory() as directory:  # 所有新数据均写入独立临时目录。
            root = Path(directory)  # 保存本次测试实验。
            old = project / "examples" / "reward-demo"  # 冻结的历史版本二样本。
            for role in ("solver", "challenger"):  # 两种旧训练契约都应保留。
                output = root / f"old-{role}.jsonl"  # 本次重新导出的历史样本。
                export_transitions(old, output, role)  # 使用正式导出器及真实回放。
                self.assertEqual(output.read_bytes(), (old / f"{role}-transitions.jsonl").read_bytes())  # 旧数据逐行与已发布版本一致。
            config = root / "config.toml"  # 设置三轮才能观察到同一快照产生两个提案。
            config.write_text((project / "config.toml").read_text().replace('"sync"', '"prefetch_one"').replace("rounds = 2", "rounds = 3").replace("historical_tasks_per_round = 0", "historical_tasks_per_round = 1"))  # 启用重测以区分 Solver 的可见输入。
            run = root / "run"  # 公共入口接管新的运行目录。
            subprocess.run([sys.executable, "-m", "bicolor", "--config", str(config), "--output", str(run)], cwd=project, check=True, capture_output=True)  # 使用离线角色执行真实调度。
            challenger, solver = root / "challenger.jsonl", root / "solver.jsonl"  # 保存新契约的双角色样本。
            export_transitions(run, challenger, "challenger")  # 重算奖励后导出真实决策。
            proposals = [json.loads(line) for line in challenger.read_text().splitlines()]  # 两个决策共用同一观察截止轮。
            self.assertEqual([row["observation"]["round"] for row in proposals], [0, 0])  # 预取第二次提案时第一轮求解尚未结束。
            self.assertEqual(len({row["decision_id"] for row in proposals}), 2)  # 两个动作可独立归属。
            for sample in proposals:  # 核对观察、动作与结算来自同一提案。
                decision = json.loads((run / f"{sample['decision_id']}.json").read_text())  # 取实际生成的提案。
                self.assertEqual((sample["observation"], sample["action"]), (decision["input"], decision["output"]))  # 不混入后来评价。
                self.assertEqual((sample["reward"], sample["challenge_reward"], sample["diagnostic_quality"]), (sample["evaluation"]["reward"], sample["evaluation"]["challenge_reward"], sample["evaluation"]["diagnostic_quality"]))  # 出题难度与诊断区分度分别保留。
            export_transitions(run, solver)  # 重测中的求解动作也正常进入样本集。
            episodes = {ep["episode_id"]: ep for ep in map(json.loads, (run / "episodes.jsonl").read_text().splitlines())}  # 按本次回合关联动作。
            rows = [json.loads(line) for line in solver.read_text().splitlines()]  # 恢复全部有效转移。
            self.assertIn("historical_retest", {row["group"] for row in rows})  # 历史回合仍获得求解训练奖励。
            for row in rows:  # 逐步与轨迹交叉核对历史内容。
                ep = episodes[row["episode_id"]]  # 用实际回合身份定位原始题目。
                obs = row["observation"]  # 模型输入与用于分析的元信息独立。
                self.assertEqual([row[k] for k in ("group", "origin_task_id", "decision_id", "solver_model")], [ep[k] for k in ("group", "origin_task_id", "decision_id", "solver_model")])  # 绑定真实组别和行为策略。
                index = ep["task"]["budget"] - obs["state"]["remaining_steps"]  # 由动作前的剩余步数求执行下标。
                self.assertEqual(obs["history"], ep["steps"][:index])  # 不包含当前动作和之后的奖励。
                self.assertEqual(row["raw_action"], ep["steps"][index]["raw_action"])  # 保存模型完整回答并另记实际翻转动作。
                self.assertEqual(set(obs), {"episode_id", "task", "state", "history", "reward_contract"})  # 不向模型泄露原题或分组标记。
