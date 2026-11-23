import io  # 构造内存中的 HTTP 响应。
import json  # 保存夹具并修改指定的一条请求。
import tempfile  # 将运行证据放在临时目录中。
import unittest  # 用标准库验证回放契约。
from pathlib import Path  # 组织临时证据路径。
from unittest.mock import patch  # 只替换远端 HTTP 服务。
from bicolor.clients import ChatClient, OfflineClient  # 运行真实请求序列化，按固定策略返回动作。
from bicolor.loop import run_rounds  # 生成旧版运行证据。
from bicolor.replay import verify_run  # 验证正式回放入口。


def answer_request(request, timeout):  # 模型协议替身接收实际传输的字节。
    body = json.loads(request.data)  # 读取序列化后的消息和模型标识。
    role = body["model"]  # 测试将角色名直接用作模型标识。
    answer = OfflineClient().ask(role, json.loads(body["messages"][1]["content"]))  # 根据真实输入运行确定性策略。
    answer["explanation"] = "合法的附加响应字段"  # Solver 执行只用节点，原始回答仍应能复核。
    return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]}).encode())  # 返回标准成功消息。


def create_run(path, version, schedule_mode="prefetch_one"):  # 通过真实调度器和 API 客户端写出一份可离线重放的实验。
    run = dict(seed=42, rounds=2, initial_size=2, completion_bonus=1.0, rollouts_per_task=1)  # 两轮就能覆盖两个角色。
    runner = run_rounds  # 使用版本二的原始执行路径。
    if version >= 3:  # 有决策身份的版本可在真实调度器中验证。
        from bicolor.curriculum import run_curriculum  # 延迟导入使旧版本独立可测。
        runner = run_curriculum  # 选择新调度器。
        run.update(rounds=3, schedule_mode=schedule_mode, historical_tasks_per_round=1)  # 第三轮能区分两个共享快照的决策。
        if version == 4:  # 新版本可以先做两轮新题，再混入历史题。
            run["historical_start_round"] = 2  # 改写该轮次应改变实际观察并导致回放失败。
    api = dict(solver_model="solver", challenger_model="challenger", max_completion_tokens=128, max_calls=240)  # 为复算保留真实请求参数。
    client = ChatClient("https://recording.invalid/v1", "unused", "solver", "challenger", path / "calls.jsonl", max_calls=240, max_completion_tokens=128)  # 测试预算只供内存响应使用。
    with patch("bicolor.clients.urlopen", side_effect=answer_request):  # 整个运行不访问网络。
        runner(client, path, **run)  # 生成实际调度、轨迹和决策证据。
    manifest = dict(schema_version=version, status="completed", evidence_mode="api", config=dict(run=dict(run, mode="api"), api=api), api_calls=client.calls)  # 记录本次启动条件。
    (path / "manifest.json").write_text(json.dumps(manifest))  # 供公开验证入口读取。


def write_calls(path, calls):  # 将受控修改后的日志重新写回测试目录。
    path.write_text("".join(json.dumps(call) + "\n" for call in calls))  # 每行仍是一条可解析的调用。


class ReplayContractTests(unittest.TestCase):  # 分别验证旧格式兼容和新格式乱序关联。
    def test_v2_raw_extra_and_full_request(self):  # 保留历史 API 动作中的额外字段并识别请求篡改。
        with tempfile.TemporaryDirectory() as directory:  # 每次运行完全隔离。
            root = Path(directory)  # 指定独立输出目录。
            create_run(root, 2)  # 生成含额外响应字段的旧格式轨迹。
            self.assertEqual(verify_run(root)["status"], "passed")  # 仅执行节点的老记录仍应通过。
            log = root / "calls.jsonl"  # 指向实际发送的请求。
            calls = [json.loads(line) for line in log.read_text().splitlines()]  # 读取证据副本。
            payload = json.loads(calls[0]["request"]["messages"][1]["content"])  # 保持回合编号和返回动作不变。
            payload["task"]["edges"] = []  # 只篡改给模型的边集。
            calls[0]["request"]["messages"][1]["content"] = json.dumps(payload)  # 保存完整但错误的题面。
            write_calls(log, calls)  # 注入请求偏差。
            with self.assertRaisesRegex(ValueError, "request"):  # 应明确拒绝实际观察不一致。
                verify_run(root)  # 通过运行环境重建输入检查。

    def test_v4_delayed_history_replay(self):  # 两种调度都必须按原启动配置复算重测回合。
        for mode in ("sync", "async"):  # 覆盖串行出题及有界题目队列。
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:  # 每种模式留下独立的录制样本。
                root = Path(directory)  # 使用临时目录保存运行。
                create_run(root, 4, mode)  # 在第三轮首次抽取一题，由本机协议替身作答。
                self.assertEqual(verify_run(root)["model_calls"], 0)  # 完整记录使用同一配置可离线重放。
                manifest_path = root / "manifest.json"  # 即使调用不变，历史题起始轮也不能被改写。
                manifest = json.loads(manifest_path.read_text())  # 读取正确配置。
                manifest["config"]["run"]["historical_start_round"] = 1  # 会在第二轮就产生一个未经记录的额外动作。
                manifest_path.write_text(json.dumps(manifest))  # 保留其余记录以识别此偏差。
                with self.assertRaises(ValueError):  # 篡改后不再能复现动作与奖励。
                    verify_run(root)  # 正式回放必须报错。

    def test_v3_reversed_calls_and_corruption(self):  # 按步和决策身份回放，不依赖网络返回顺序。
        with tempfile.TemporaryDirectory() as directory:  # 使用独立的三轮预取运行。
            root = Path(directory)  # 指向测试证据目录。
            create_run(root, 3)  # 新格式保存全量原始动作。
            log = root / "calls.jsonl"  # 所有 HTTP 请求和响应在此记录。
            calls = [json.loads(line) for line in reversed(log.read_text().splitlines())]  # 强制倒置整个日志的完成顺序。
            write_calls(log, calls)  # 刻意改变行顺序。
            self.assertEqual(verify_run(root)["status"], "passed")  # 不同完成顺序应得到相同奖励归属。
            for field in ("model", "answer", "feedback", "duplicate"):  # 分别检查模型、响应、快照及重复身份。
                with self.subTest(field=field):  # 报告具体没有被识别的篡改。
                    changed = json.loads(json.dumps(calls))  # 每个场景从同一份正确日志开始。
                    role = "solver" if field == "answer" else "challenger"  # 检查新版合法 Solver 动作也能保留原始附加字段。
                    call = next(item for item in changed if item["role"] == role)  # 保持当前回合步或决策身份一致。
                    if field == "model":  # 角色标签正确但实际模型错误。
                        call["request"]["model"] = "other"  # 模拟路由错配。
                    elif field == "answer":  # 保持执行动作但改写原始回答中的附加说明。
                        message = call["response"]["choices"][0]["message"]  # 取得原始模型回复。
                        message["content"] = json.dumps(dict(json.loads(message["content"]), explanation="伪造"))  # 不能以执行结果相同替代原始回答一致。
                    elif field == "feedback":  # 决策编号不变但观察不同。
                        message = call["request"]["messages"][1]  # 取得发送给 Challenger 的用户输入。
                        message["content"] = json.dumps(dict(json.loads(message["content"]), feedback=[]))  # 改写决策所见的反馈。
                    else:  # 同一回放键不应被消费两次。
                        changed[0] = changed[1]  # 同时引入一个重复调用和一个缺失调用。
                    write_calls(log, changed)  # 写入当前的错误样本。
                    with self.assertRaises(ValueError):  # 任一外部证据错位均应失败。
                        verify_run(root)  # 验证真实入口。
