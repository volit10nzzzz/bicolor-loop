import argparse  # 提供最小命令行入口。
import json  # 保存运行状态与摘要。
import os  # 从环境变量读取密钥。
from pathlib import Path  # 管理实验目录。
import tomllib  # 读取标准 TOML 配置。
from .clients import ChatClient, OfflineClient  # 选择真实接口或明确的离线替身。
from .curriculum import run_curriculum  # 调用真实算法链路。


def main():  # 启动一次完整实验。
    parser = argparse.ArgumentParser(description="冻结模型的二染色 Solver–Challenger 循环")  # 描述运行目的。
    parser.add_argument("--config", default="config.toml")  # 默认读取项目配置。
    parser.add_argument("--mode", choices=("offline", "api"))  # 允许显式覆盖模式。
    parser.add_argument("--output", required=True)  # 要求独立输出目录，避免覆盖证据。
    args = parser.parse_args()  # 解析用户参数。
    config = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))  # 读取配置内容。
    run, api = config["run"], config["api"]  # 区分环境参数与模型参数。
    mode = args.mode or run["mode"]  # 命令行优先于配置。
    if mode not in ("offline", "api"):  # 校验外部配置边界。
        parser.error("mode must be offline or api")  # 不静默降级成离线模式。
    key = os.environ.get(api["key_env"], "")  # 密钥不进入配置副本。
    if mode == "api" and (not key or any(not api[name] or api[name].startswith("SET_") for name in ("solver_model", "challenger_model"))):  # 启动前检查必需配置。
        parser.error("API mode requires selected models and the configured API-key environment variable")  # 不发出注定失败的请求。
    output = Path(args.output)  # 使用明确指定的实验位置。
    output.mkdir(parents=True, exist_ok=False)  # 已存在目录必须由用户另起名称。
    client = OfflineClient() if mode == "offline" else ChatClient(api["base_url"], key, api["solver_model"], api["challenger_model"], output / "calls.jsonl", api["max_calls"], api["max_completion_tokens"], api["timeout_seconds"])  # 建立单个共享预算客户端。
    manifest = {"schema_version": 4, "status": "running", "evidence_mode": client.evidence_mode, "config": config}  # 记录配置，明确没有权重更新。
    manifest["config"]["run"]["mode"] = mode  # 保存实际生效模式。
    try:  # 保留失败运行的已完成轨迹。
        schedule_mode = run.setdefault("schedule_mode", "sync")  # 旧配置自动采用同步调度。
        history_count = run.setdefault("historical_tasks_per_round", 0)  # 缺省不增加重测请求。
        start_round = run.setdefault("historical_start_round", 1)  # 从第几批开始加入重测，轮号由零开始。
        records = run_curriculum(client, output, run["seed"], run["rounds"], run["rollouts_per_task"], run["initial_size"], run["completion_bonus"], schedule_mode, history_count, start_round)  # 按配置执行内外循环。
        fields = ("round", "summary", "prediction_results", "challenger_reward", "coverage", "monitor", "retests")  # 公开奖励及分层监控。
        summary = [{key: item[key] for key in fields} for item in records]  # 汇总可复核结果。
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")  # 提供简洁实验摘要。
        manifest["status"] = "completed"  # 仅完整返回才标记成功。
        print(json.dumps({"output": str(output.resolve()), "evidence_mode": client.evidence_mode, "rounds": len(records)}, ensure_ascii=False))  # 输出可定位的实验结果。
    except Exception as error:  # 不隐藏接口或算法链路失败。
        manifest.update(status="error", error_type=type(error).__name__)  # 保留失败类型，避免记录凭据。
        raise  # 保留非零退出状态。
    finally:  # 成功和失败都保存状态。
        manifest["api_calls"] = getattr(client, "calls", 0)  # 离线运行没有真实请求。
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")  # 写入实验清单。


if __name__ == "__main__":  # 支持 python -m bicolor。
    main()  # 调用入口。
