import json  # 读取调度证据。
from pathlib import Path  # 管理隔离目录。
from tempfile import TemporaryDirectory  # 隔离试验文件。
from threading import Event  # 用因果信号验证并发，而非计时猜测。
import unittest  # 使用标准测试断言。
from bicolor.clients import OfflineClient  # 保留确定性模型边界。
from bicolor.curriculum import run_curriculum  # 执行正式调度器。
from bicolor.graph import generate_task  # 建立需要三步的真实初始状态。
from bicolor.loop import solve  # 在两个求解动作之间注入预取失败。


class OverlapClient(OfflineClient):  # 只有两角色调用确实重叠时才放行。
    def __init__(self):  # 记录两个到达点。
        self.solver = Event()  # 第一轮 Solver 到达事件。
        self.producer = Event()  # 第二批预取任务到达事件。

    def ask(self, role, payload):  # 复用离线策略并检查时间依赖。
        if role == "challenger" and payload["planned_round"] == 2:  # 为第2轮出题时必须只看到 H0。
            self.producer.set()  # 通知求解线程已经开始准备。
            if not self.solver.wait(3):  # 单线程串行将触发超时。
                raise AssertionError("solver did not overlap")  # 检测串行执行。
        if role == "solver" and payload["episode_id"] == "r1-q0-e0" and not payload["history"]:  # Q1首步必须与准备Q2重叠。
            self.solver.set()  # 通知出题线程。
            if not self.producer.wait(3):  # 验证另一个方向的因果关系。
                raise AssertionError("producer did not overlap")  # 检测生产者未启动。
        return super().ask(role, payload)  # 真实状态转移在调度器内执行。


class ScheduleTests(unittest.TestCase):  # 校验观测截止点与历史重测边界。
    def test_prefetch_failure_keeps_executed_step(self):  # 后台出题失败时已执行动作仍须落入中断轨迹。
        saved = []  # 收集真实中断回调。
        checks = iter([True, False])  # 首步时生产者仍在运行，次步前已失败。
        def check():  # 重现调度器传播的异常。
            if not next(checks):  # 第一个动作已执行。
                raise RuntimeError("prefetch failed")  # 阻止第二次求解调用。
        with self.assertRaisesRegex(RuntimeError, "prefetch failed"):  # 失败不能被求解器吞掉。
            solve(generate_task("bridge", 2, 42, "q"), OfflineClient(), "e", saved.append, keep_action=True, before_step=check)  # 走真实求解循环。
        self.assertEqual([(e["status"], len(e["steps"])) for e in saved], [("interrupted", 1)])  # 保留第一步和中断状态。

    def test_prefetch_snapshot_and_retests(self):  # 首次重用 H0 时不能把两个决策混为一谈。
        with TemporaryDirectory() as directory:  # 最小三轮才会预取。
            out = Path(directory)  # 保存独立运行。
            rows = run_curriculum(OverlapClient(), out, rounds=3, rollouts_per_task=1, schedule_mode="prefetch_one", historical_tasks_per_round=5)  # 故意请求超过候选数。
            a, b = [json.loads((out / f"decision-{n}.json").read_text()) for n in (0, 1)]  # 查两份出题证据。
            self.assertEqual((a["observation_round"], b["observation_round"]), (0, 0))  # 两次都只看首轮。
            self.assertNotEqual(a["decision_id"], b["decision_id"])  # 两个独立动作。
            self.assertEqual(rows[2]["challenger_reward"]["decision_id"], b["decision_id"])  # 给生成Q2的动作计奖。
            self.assertEqual(rows[2]["challenger_reward"]["target_episodes"], 2)  # 两个同题型重测回合不能混入。
            originals = {task["task_id"]: task for task in rows[0]["tasks"] if task["group"] == "target"}  # 只取截止时间前的目标题。
            for row in rows[1:]:  # 两个预取批次共享候选截止时间。
                historical = [task for task in row["tasks"] if task["group"] == "historical_retest"]  # 显式分组。
                self.assertCountEqual([task["origin_task_id"] for task in historical], list(originals))  # 每道原题只抽一次。
                for task in historical:  # 重测保留原始题面。
                    original = originals[task["origin_task_id"]]  # 对应原始任务。
                    for key in ("edges", "colors", "budget"):  # 复核图、着色与预算。
                        self.assertEqual(task[key], original[key])  # 禁止重新随机化。
            self.assertIsNone(b["input"]["feedback"]["challenger_reward"])  # H0尚无模型提案结算。

    def test_sync_feedback(self):  # 同步决策应立即看到上一提案已结算的反馈。
        with TemporaryDirectory() as directory:  # 零重测的默认路径。
            out = Path(directory)  # 保存可检查的数据。
            rows = run_curriculum(OfflineClient(), out, rounds=3, rollouts_per_task=1)  # 同步三轮。
            b = json.loads((out / "decision-1.json").read_text())  # 输入用于出Q2。
            self.assertEqual(b["observation_round"], 1)  # 可见H1。
            self.assertEqual(b["input"]["feedback"]["challenger_reward"], rows[1]["challenger_reward"])  # 已结算反馈完全一致。
            self.assertEqual([len(row["episodes"]) for row in rows], [3, 3, 3])  # 默认不增加历史回合。
