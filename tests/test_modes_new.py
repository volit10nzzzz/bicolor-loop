import json  # 检查实际交给出题方的观察。
from pathlib import Path  # 为每次运行分配独立证据目录。
from tempfile import TemporaryDirectory  # 测试不修改示例轨迹。
from threading import Event, Thread  # 用交接信号控制并发，限制故障测试的等待。
import unittest  # 只使用标准库测试框架。
from bicolor.clients import OfflineClient  # 模拟模型，保留真实环境及奖励计算。
from bicolor.curriculum import run_curriculum  # 走正式训练前的调度入口。


class RendezvousClient(OfflineClient):  # 在解第一批与准备第二批相遇时控制请求返回。
    def __init__(self, fail_role=None):  # 可选择在两个并发参与者中的一方触发错误。
        self.fail_role = fail_role  # 空值代表两方正常提交响应。
        self.solver_started = Event()  # 消费者进入当前题的首步。
        self.challenger_started = Event()  # 生产者进入下一批的提案请求。
        self.next_batch_consumed = Event()  # 观察故障后是否仍然求解第二批。

    def ask(self, role, payload):  # 每个事件只对应一次指定的真实调用。
        if role == "solver" and payload["episode_id"].startswith("r2-"):  # 只关心第二批题有没有实际发给解题方。
            self.next_batch_consumed.set()  # 不依赖队列内部的落盘时机。
        if role == "challenger" and payload["planned_round"] == 2:  # 为第二批出题的生产者必须等到求解已开始。
            self.challenger_started.set()  # 明确通知正在等题的测试客户端。
            if not self.solver_started.wait(3):  # 串行执行无法相遇。
                raise AssertionError("missing concurrent solver")  # 诊断生产早于求解而被阻塞。
        elif role == "solver" and payload["episode_id"] == "r1-q0-e0" and not payload["history"]:  # 第一批第一题首步。
            self.solver_started.set()  # 放行正在等待的生产者。
            if not self.challenger_started.wait(3):  # 后台提案必须独立推进。
                raise AssertionError("missing concurrent challenger")  # 诊断消费等待出题串行完成。
        else:  # 其它请求始终由普通离线策略响应。
            return super().ask(role, payload)  # 不改变算法行为。
        if role == self.fail_role:  # 在双方已进入各自推理之后精确注入故障。
            raise RuntimeError(f"failed {role}")  # 运行器应当把原错误传回调用方。
        return super().ask(role, payload)  # 正常完成受控的并发请求。


class ModeTests(unittest.TestCase):  # 检查两种调度在同样的历史重测配置下的实际行为。
    def test_retests_start_at_configured_round(self):  # 被重测的是旧题面，由本轮重新作答和分组。
        for mode, cutoffs in (("sync", [None, 0, 1, 2]), ("async", [None, 0, 0, 1])):  # 两种模式的可见证据不同。
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:  # 使用独立运行，避免串用记录。
                out = Path(tmp)  # 保存真实决策文件。
                client = RendezvousClient() if mode == "async" else OfflineClient()  # 异步时要求生产和消费实际重叠。
                rows = run_curriculum(client, out, rounds=4, rollouts_per_task=1, schedule_mode=mode, historical_start_round=2, historical_tasks_per_round=3)  # 从第二轮起每轮申请三题。
                self.assertEqual([row["observation_round"] for row in rows], cutoffs)  # 先验算每份题单的快照截止点。
                for index, row in enumerate(rows):  # 将每一次随机选择与当时已存在的原题比较。
                    historical = [t for t in row["tasks"] if t["group"] == "historical_retest"]  # 同族的目标题和重测题必须可区分。
                    if index < 2:  # 开关在编号为二的批次起效。
                        self.assertEqual(historical, [])  # 此前即使有原题候选也不启动重测。
                        continue  # 首批无模型决策可验。
                    cutoff = cutoffs[index]  # 以真实可见信息为索引边界。
                    originals = {t["task_id"]: t for r in rows[:cutoff + 1] for t in r["tasks"] if t["group"] == "target"}  # 对照和重测均排除。
                    origins = [t["origin_task_id"] for t in historical]  # 同批按原题编号检查重复。
                    self.assertEqual(len(origins), min(3, len(originals)))  # 异步第二批仅可用种子中的两道原题。
                    self.assertEqual(len(set(origins)), len(origins))  # 不重复采样同一道原题。
                    self.assertTrue(set(origins).issubset(originals))  # 不从当前或未来批次借题。
                    for t in historical:  # 不允许只沿用题型和种子标签而重建另一张图。
                        origin = originals[t["origin_task_id"]]  # 对照首次原题。
                        self.assertEqual((t["edges"], t["colors"], t["budget"]), (origin["edges"], origin["colors"], origin["budget"]))  # 验算旧题的公开题面。
                        self.assertIsNone(t["decision_id"])  # 此前的题不应替当前提案获奖。
                    retest_ids = {t["task_id"] for t in historical}  # 新回合必须真正进入 Solver 作答结果。
                    replayed = [e["task_id"] for e in row["episodes"] if e["group"] == "historical_retest"]  # 检查每个新回合的来源。
                    self.assertCountEqual(replayed, retest_ids)  # 每道抽样的旧题都被本轮独立求解一次。
                    self.assertEqual(row["prediction_results"]["target"]["planned"], 2)  # 每份提案仍只有两道目标题。
                    self.assertEqual(row["challenger_reward"]["target_episodes"], 2)  # 历史重测不改变有效挑战分母。
                    observation = json.loads((out / f"{row['decision_id']}.json").read_text())["input"]  # 读取实际提交给 Challenger 的数据。
                    self.assertEqual([e["episode_id"] for e in observation["episodes"]], [e["episode_id"] for e in rows[cutoff]["episodes"]])  # 断言快照里的真实轨迹集合。
                    self.assertEqual(observation["feedback"]["decision_id"], rows[cutoff]["decision_id"])  # 防止提案读取未来已结算奖励。

    def test_retest_choice_uses_run_seed(self):  # 验证可复算的随机抽题，也能区分每次固定取最早两题的实现。
        for mode in ("sync", "async"):  # 两种模式都应支持种子重现。
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:  # 每次执行使用独立证据目录。
                picks = []  # 保存候选池充足时实际抽中的原题编号。
                for run, seed in enumerate((42, 42, 43, 44, 45)):  # 相同种子重复两次，另有三个不同种子。
                    out = Path(tmp) / str(run)  # 不覆盖上一轮的输出文件。
                    rows = run_curriculum(OfflineClient(), out, seed=seed, rounds=5, rollouts_per_task=1, schedule_mode=mode, historical_start_round=4, historical_tasks_per_round=2)  # 第四轮已有多于两题可抽。
                    picks.append(frozenset(t["origin_task_id"] for t in rows[4]["tasks"] if t["group"] == "historical_retest"))  # 比较题目而非题单顺序。
                self.assertEqual(picks[0], picks[1])  # 同种子、同快照必须能还原抽样。
                self.assertGreater(len(set(picks)), 1)  # 多个种子至少给出两种不同原题集合。

    def test_retests_are_off_by_default(self):  # 用户未打开重测时，两种调度都不应增加求解回合。
        for mode in ("sync", "async"):  # 同步和队列各走一次。
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:  # 互相隔离题单。
                rows = run_curriculum(OfflineClient(), Path(tmp), rounds=3, rollouts_per_task=1, schedule_mode=mode)  # 故意省略重测参数。
                self.assertEqual([len(r["episodes"]) for r in rows], [3, 3, 3])  # 只执行每轮两目标加一对照。

    def test_async_errors_cross_queue_boundary(self):  # 对方退出后，等待题或等待下一次观察的一侧必须能结束。
        for role in ("challenger", "solver"):  # 覆盖方向相反的两次故障。
            with self.subTest(role=role), TemporaryDirectory() as tmp:  # 不让失败运行留下产物。
                client = RendezvousClient(role)  # 指定哪一侧中断。
                errors = []  # 把调用线程中的原始错误带回测试线程。
                def run():  # 设定等待上限，队列死锁时测试能明确失败。
                    try:  # 只有注入错误才是预期退出。
                        run_curriculum(client, Path(tmp), rounds=4, rollouts_per_task=1, schedule_mode="async")  # 使用正常的生产、取题和错误传播入口。
                    except Exception as error:  # 保留真实错误类型与消息供断言区分。
                        errors.append(error)  # 错误不可被替换为普通的运行完成。
                thread = Thread(target=run, daemon=True)  # 故障测试不会无限阻塞主测试线程。
                thread.start()  # 开始失败场景。
                thread.join(6)  # 正常退出无需等到超时。
                self.assertFalse(thread.is_alive(), "queue did not stop")  # 排查任一方遗留的阻塞。
                self.assertEqual([(type(e), str(e)) for e in errors], [(RuntimeError, f"failed {role}")])  # 两端保留原始异常。
                self.assertFalse(client.next_batch_consumed.is_set())  # 本批失败后不应执行下一批。
