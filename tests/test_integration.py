import json  # 检查真实序列化边界。
import os  # 为本机服务隔离代理设置。
from unittest.mock import patch  # 仅覆盖测试进程的代理绕过列表。
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # 建立本地可控接口。
from pathlib import Path  # 管理临时实验文件。
import subprocess  # 验证真实命令行入口。
import sys  # 使用当前 Python 解释器。
import tempfile  # 隔离测试输出。
import threading  # 在后台服务本地 HTTP 请求。
import unittest  # 使用标准库测试框架。
from bicolor.clients import ChatClient, OfflineClient  # 仅替换远端服务。
from bicolor.loop import run_rounds  # 保留真实内外循环。
from bicolor.train import export_transitions  # 验证训练数据契约。
from bicolor.replay import verify_run  # 核对 HTTP 记录与奖励复算。


class Handler(BaseHTTPRequestHandler):  # 用离线策略提供符合协议的 HTTP 响应。
    malformed = False  # 允许测试模型返回非 JSON 的错误路径。

    def do_POST(self):  # 接收真实客户端发出的请求。
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))  # 解析实际请求体。
        role = "solver" if body["model"] == "solver-test" else "challenger"  # 验证模型路由。
        answer = OfflineClient().ask(role, json.loads(body["messages"][1]["content"]))  # 用真实观察生成替身响应。
        content = "not-json" if self.malformed else json.dumps(answer)  # 注入解析失败场景。
        result = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()  # 构造协议响应。
        self.send_response(200)  # 返回成功的 HTTP 状态。
        self.send_header("Content-Type", "application/json")  # 指定响应类型。
        self.end_headers()  # 完成 HTTP 头部。
        self.wfile.write(result)  # 写入真实网络流。

    def log_message(self, format, *args):  # 测试不输出无关访问日志。
        pass  # 保持测试报告简洁。


class IntegrationTests(unittest.TestCase):  # 覆盖命令行、HTTP 和训练入口。
    def test_cli_and_training_export(self):  # 完整离线运行后检查转移终止语义。
        with tempfile.TemporaryDirectory() as directory:  # 创建隔离目录。
            root = Path(directory)  # 统一文件操作。
            run = root / "run"  # CLI 要求目录尚不存在。
            result = subprocess.run([sys.executable, "-m", "bicolor", "--output", str(run)], capture_output=True, text=True)  # 执行公开启动命令。
            self.assertEqual(result.returncode, 0, result.stderr)  # CLI 应成功退出。
            report = export_transitions(run, root / "transitions.jsonl")  # 调用同一导出逻辑。
            rows = [json.loads(line) for line in (root / "transitions.jsonl").read_text().splitlines()]  # 检查真实输出。
            self.assertEqual(sum(row["terminated"] for row in rows), 12)  # 两轮三题各两次采样都有终局。
            self.assertFalse(report["optimizer_started"])  # 导出不应启动训练。
            self.assertEqual(report["transitions"], len(rows))  # 统计应与文件一致。
            self.assertTrue(all(row["evidence_mode"] == "offline_simulation" for row in rows))  # 保留模拟标识。
            again = subprocess.run([sys.executable, "-m", "bicolor", "--output", str(run)], capture_output=True)  # 尝试覆盖原运行。
            self.assertNotEqual(again.returncode, 0)  # 不能覆盖已经保存的证据。

    def test_http_loop_budget_and_bad_response(self):  # 本地服务验证 HTTP 适配而非真实模型质量。
        with patch.dict(os.environ, {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}), tempfile.TemporaryDirectory() as directory, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:  # 本机协议验证不经外部代理。
            thread = threading.Thread(target=server.serve_forever, daemon=True)  # 在独立线程处理请求。
            thread.start()  # 启动本地服务。
            try:  # 确保任何断言失败后都停止服务。
                root = Path(directory)  # 使用临时输出目录。
                client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "secret-test-token", "solver-test", "challenger-test", root / "calls.jsonl")  # 使用真实 HTTP 客户端。
                records = run_rounds(client, root, rollouts_per_task=1)  # 跨 HTTP 跑完整两轮。
                manifest = {"schema_version": 2, "status": "completed", "evidence_mode": "api", "api_calls": client.calls, "config": {"run": {"mode": "api", "seed": 42, "rounds": 2, "rollouts_per_task": 1, "initial_size": 2, "completion_bonus": 1.0}}}  # 本机协议夹具，仅用于接口验证。
                (root / "manifest.json").write_text(json.dumps(manifest))  # 提供正式回放所需运行参数。
                self.assertEqual(verify_run(root)["status"], "passed")  # 真实 HTTP 记录也应能无网络复算。
                self.assertEqual(records[1]["challenger_reward"]["reward"], 1.0)  # 出题奖励通过 HTTP 链路结算。
                self.assertTrue(any(step["completion_bonus"] == 1 for record in records for episode in record["episodes"] for step in episode["steps"]))  # 至少一次成功触发完成加成。
                logs = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]  # 读取调用记录。
                self.assertEqual(sum(item["role"] == "challenger" for item in logs), 1)  # 两轮之间恰好调用一次出题器。
                self.assertEqual(records[1]["proposal"], records[0]["next_proposal"])  # HTTP 输出实际控制下一轮。
                self.assertNotIn("secret-test-token", (root / "calls.jsonl").read_text())  # 授权密钥不能进入日志。
                client.max_calls = client.calls  # 将预算限制为已用额度。
                with self.assertRaisesRegex(RuntimeError, "budget"):  # 下一次请求应在网络前被拒绝。
                    client.ask("solver", {})  # 空内容也不能绕过预算。
                client.max_calls += 1  # 为协议错误测试明确增加一次本地额度。
                Handler.malformed = True  # 返回格式损坏的模型内容。
                with self.assertRaises(json.JSONDecodeError):  # 解析错误必须暴露。
                    client.ask("solver", {"state": {"colors": [0, 0]}, "task": {"edges": [[0, 1]]}})  # 发送可解析请求。
                last = json.loads((root / "calls.jsonl").read_text().splitlines()[-1])  # 检查失败日志。
                self.assertEqual(last["status"], "error")  # 不能伪造成功响应。
            finally:  # 恢复测试共享状态并释放端口。
                Handler.malformed = False  # 防止影响后续测试。
                server.shutdown()  # 停止服务循环。
                thread.join()  # 等待线程退出。


if __name__ == "__main__":  # 支持单独运行。
    unittest.main()  # 执行集成检查。
