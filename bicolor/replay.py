import argparse  # 提供无网络回放命令。
import json  # 解析原始请求、动作及运行证据。
from pathlib import Path  # 定位实验文件。
import tempfile  # 将重跑限制在独立目录中。
from . import clients  # 复用真实客户端的角色提示。
from .loop import run_rounds  # 按原语义重放旧版轮次。


def read_json(path):  # 读取一个 JSON 证据文件。
    return json.loads(path.read_text(encoding="utf-8"))  # 使用与写入端相同的编码。


def read_rows(path):  # 读取逐行保存的轨迹或请求。
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]  # 空行或截断记录均应报错。


def same(left, right):  # 在 JSON 语义下比较嵌套结构，避免把布尔值误认成节点整数。
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)  # 同时归一化键序及内存中的元组。


def action_key(role, payload, version):  # 两个角色分别使用执行时可见的身份。
    if role == "solver":  # 每个 Solver 回合可以调用多次。
        return role, payload["episode_id"], len(payload["history"])  # 当前步号由此前已执行的步数给出。
    if role == "challenger":  # 预取可能对同一轮快照出题两次。
        return role, payload["decision_id"] if version >= 3 else payload["round"]  # 旧版同步调度每轮仅出一次题。
    raise ValueError("unknown API request role")  # 不接受没有关联规则的角色。


class ReplayClient:  # 用真实环境逐步重建输入，仅以保存的动作替代模型。
    def __init__(self, run_dir, manifest, records):  # 按业务身份建索引，与 HTTP 调用完成顺序无关。
        self.version = manifest["schema_version"]  # 选择相应的历史数据契约。
        self.evidence_mode = manifest["evidence_mode"]  # 在临时目录复现原标签，对外报告另标记为回放。
        self.api = manifest["config"].get("api", {})  # 部分旧版协议夹具没有独立保存模型配置。
        self.models = dict(clients.OfflineClient.models) if self.evidence_mode == "offline_simulation" else {}  # 离线复用原策略名称，真实接口的模型从运行清单绑定。
        self.actions = {}  # 保存轨迹声明的动作、以及是否仅保留了合法节点。
        self.requests = {}  # API 证据另按业务键保存，离线来源无需构造请求。
        for record in records:  # 从保存的轨迹取得模型动作。
            for episode in record["episodes"]:  # 每个回合独立从第零步开始。
                for index, step in enumerate(episode["steps"]):  # 记录每个回合的动作次序。
                    node_only = self.version == 2 and step["valid"]  # 旧版仅在合法动作时丢弃附加字段。
                    if self.version >= 3:  # 新版必须提供原始动作。
                        if "raw_action" not in step:  # 缺少完整回答时不得假定它只包含节点。
                            raise ValueError(f"missing raw_action in v{self.version} episode")  # 明确指出证据不足。
                        answer = step["raw_action"]  # 包含模型输出的所有字段。
                    else:  # 老格式按当时实际保存的信息核验。
                        answer = {"node": step["node"]} if node_only else step["action"]  # 无效动作在旧版也有原始记录。
                    self.add(("solver", episode["episode_id"], index), answer, node_only)  # 按回合和步绑定。
        for path in run_dir.glob("decision-*.json"):  # 每个 Challenger 动作有独立的证据文件。
            decision = read_json(path)  # 同一观察轮可能产生不同的决策。
            self.add(action_key("challenger", decision["input"], self.version), decision["output"], False)  # 不以文件顺序推断动作。
        if self.evidence_mode == "api":  # 网络证据需要额外绑定请求和返回值。
            self.load_calls(run_dir, manifest)  # 在不访问服务的情况下解析所有记录。
        elif manifest["api_calls"] != 0 or (run_dir / "calls.jsonl").exists() and read_rows(run_dir / "calls.jsonl"):  # 离线运行不能带有真实请求。
            raise ValueError("offline evidence reports API calls")  # 拒绝与原始来源声明相冲突的数据。

    def add(self, key, answer, node_only):  # 一个语义动作只能在轨迹中出现一次。
        if key in self.actions:  # 重复身份导致奖励无法惟一归属。
            raise ValueError("duplicate recorded action")  # 即便答案相同也拒绝。
        self.actions[key] = answer, node_only  # 保存可核验的原始回答或旧版合法节点。

    def load_calls(self, run_dir, manifest):  # 将历史传输记录与确定性的执行键绑定。
        calls = read_rows(run_dir / "calls.jsonl")  # 跳过网络，直接读取实际传输证据。
        if len(calls) != manifest["api_calls"] or len(calls) > self.api.get("max_calls", len(calls)):  # 检查共享预算和记账总量。
            raise ValueError("API call count differs from manifest or budget")  # 错漏计数均不能通过。
        if sorted(call["call"] for call in calls) != list(range(1, len(calls) + 1)):  # 即使乱序也应能恢复连续发出的编号。
            raise ValueError("API request sequence has missing or duplicate identities")  # 不以行号替代真实编号。
        for role in ("solver", "challenger"):  # 按清单中实际保存的配置独立绑定角色模型。
            if self.version >= 3 or f"{role}_model" in self.api:  # 新版要求完整配置，历史版本按已记录字段核验。
                self.models[role] = self.api[f"{role}_model"]  # 防止错路由的记录冒充正确模型。
        for call in calls:  # HTTP 返回行顺序不影响实际调度。
            role, request = call["role"], call["request"]  # 保存原日志的角色和请求体。
            payload = json.loads(request["messages"][1]["content"])  # 读取模型收到的完整观察。
            key = action_key(role, payload, self.version)  # 对照此次调用的回合步或决策。
            if self.version >= 3 and call.get("request_id") != f"request-{call['call']}":  # 新版给每次实际请求分配稳定的编号。
                raise ValueError("API request identity differs from call number")  # 不能用行号替换缺失的请求 ID。
            choice = call["response"]["choices"][0]  # 成功运行每次都应产出一项可解析的回答。
            if call["status"] != "ok" or choice.get("finish_reason") == "length" or key in self.requests:  # 不接受协议失败或重复效果。
                raise ValueError("failed or duplicate API request")  # 避免后来的响应覆盖原证据。
            if self.version == 2:  # 兼容未完整保存模型配置的早期清单。
                self.models.setdefault(role, request["model"])  # 仍可核对同一角色各次请求的模型是否一致。
            self.requests[key] = request, json.loads(choice["message"]["content"])  # 保留原始请求体和未裁剪的回答。
        if self.requests.keys() != self.actions.keys():  # 请求总量相同也可能存在语义错配。
            raise ValueError("API requests differ from recorded action identities")  # 检查逐个动作均有对应的证据。

    def ask(self, role, payload):  # 由实际调度器调用，payload 是当前步重新构造的观察。
        key = action_key(role, payload, self.version)  # 与历史请求使用相同的业务定位方式。
        if key not in self.actions:  # 调度若重复或新增一步则无法回放。
            raise ValueError("missing or already consumed recorded action")  # 不尝试补造动作。
        answer, node_only = self.actions.pop(key)  # 与其他并发请求消费不同的唯一动作。
        if self.evidence_mode == "api":  # 对比实际发送的提示、模型、角色及全部用户输入。
            request, raw = self.requests[key]  # 原始回答不受旧版动作字段裁剪影响。
            prompt = clients.SOLVER_PROMPT if role == "solver" else clients.CHALLENGER_PROMPT  # 旧版始终使用原提示。
            if role == "challenger" and self.version >= 3:  # 新版 Challenger 接收显式反馈。
                prompt = clients.CHALLENGER_PROMPT_V3  # 指令与新运行的实际提示一致。
            expected = {  # 重建完整的传输请求，不忽略未知的附加指令。
                "model": self.models[role],  # 对照配置指定的角色模型。
                "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": payload}],  # 保留角色顺序和整个决策观察。
                "max_completion_tokens": self.api["max_completion_tokens"] if self.version >= 3 else self.api.get("max_completion_tokens", request["max_completion_tokens"]),  # 极简旧清单仅能检查当时保存的字段。
            }  # 以重跑请求对照原始传输。
            actual = dict(request, messages=[dict(message) for message in request["messages"]])  # 保留实际消息的角色、次序及附加字段。
            actual["messages"][1]["content"] = json.loads(actual["messages"][1]["content"])  # 以完整用户对象比较，忽略 JSON 空白差异。
            if not same(actual, expected):  # 保持身份不变仍不能替换题面、历史或反馈。
                raise ValueError(f"API request differs from replay for {key}")  # 报错只暴露实验内的关联编号。
            if not isinstance(raw, dict) or not same({"node": raw.get("node")} if node_only else raw, answer):  # 旧版合法动作只对照当时保存的节点。
                raise ValueError(f"API response differs from recorded action for {key}")  # 完整原始动作不得任意删改。
            return raw  # 把实际收到的回答送入真实环境验证执行结果。
        return answer  # 离线回放直接消费已保存的策略动作。


def verify_run(run_dir):  # 不调用模型，复算已完成的历史运行。
    manifest = read_json(run_dir / "manifest.json")  # 获取版本和当时生效的配置。
    version = manifest.get("schema_version")  # 老版与新版有不同的调度契约。
    if version not in (2, 3, 4) or manifest["status"] != "completed":  # 未完成的运行单独留作诊断。
        raise ValueError("verification requires a completed schema_version=2, 3 or 4 run")  # 只报告完整可复现的运行。
    records = read_rows(run_dir / "rounds.jsonl")  # 原始逐轮证据在回放后再完整比对。
    run = manifest["config"]["run"]  # 模型以外的参数均来自启动清单。
    source = {"offline": "offline_simulation", "api": "api"}.get(run["mode"])  # 模式决定来源标签。
    if source is None or manifest["evidence_mode"] != source or any(item["evidence_mode"] != source for item in records):  # 对照清单与所有轮次。
        raise ValueError("evidence source differs between config, manifest and rounds")  # 不允许只改标签冒充接口数据。
    client = ReplayClient(run_dir, manifest, records)  # 原始证据只读。
    params = {key: run[key] for key in ("seed", "rounds", "rollouts_per_task", "initial_size", "completion_bonus")}  # 使用旧版和新版共同参数。
    runner = run_rounds  # 默认按旧版的同步调度与历史字段重放。
    if version >= 3:  # 新版必须经过真实预取、重测和分组奖励链路。
        from .curriculum import run_curriculum  # 旧版独立回放无需引入新调度器。
        runner = run_curriculum  # 选择与清单版本一致的运行器。
        params.update(schedule_mode=run.get("schedule_mode", "sync"), historical_tasks_per_round=run.get("historical_tasks_per_round", 0))  # 复用 CLI 允许的默认设置。
        if version == 3 and params["schedule_mode"] not in ("sync", "prefetch_one"):  # 不把新版模式倒标为历史版本。
            raise ValueError("invalid schedule mode for schema_version=3")  # 旧调度器只接受同步与单批预取。
        if version == 4:  # 版本四按当次清单重建开始混入历史题的轮次。
            params["historical_start_round"] = run.get("historical_start_round", 1)  # 缺省沿用第一批非种子题即可重测的约定。
    with tempfile.TemporaryDirectory() as directory:  # 不写入既有实验。
        output = Path(directory)  # 隔离重跑产生的所有文件。
        replayed = runner(client, output, **params)  # 重新生成任务、观察、动作执行后果、监控及奖励。
        if client.actions or not same(replayed, records):  # 确认所有动作均被消费且轮次内容逐项一致。
            raise ValueError("recorded states, rewards or decisions do not match replay")  # 拒绝部分匹配。
        if not same(read_rows(output / "episodes.jsonl"), read_rows(run_dir / "episodes.jsonl")):  # 独立逐回合日志也由导出方使用。
            raise ValueError("episode file differs from replay")  # 防止训练导出读取未核验副本。
        for pattern in ("decision-*.json", "schedule-*.json"):  # 比较决策快照及实际题目分配。
            expected = {path.name for path in output.glob(pattern)}  # 由重新执行的真实调度器确定应有文件。
            if expected != {path.name for path in run_dir.glob(pattern)}:  # 多出的旧决策同样可能污染导出。
                raise ValueError(f"recorded file set differs for {pattern}")  # 不能忽略额外或缺失的关联记录。
            for name in expected:  # 文件顺序不代表提案与回合的所属关系。
                if not same(read_json(output / name), read_json(run_dir / name)):  # 逐项核对求解前状态及出题观察。
                    raise ValueError(f"{name} differs from replay")  # 指出第一个不匹配的证据文件。
    if (run_dir / "summary.json").exists():  # CLI 额外保存摘要，直接调用运行器的历史夹具没有此文件。
        fields = ("round", "summary", "prediction_results", "challenger_reward", "coverage")  # 旧版摘要字段。
        if version >= 3:  # 新版还汇总分组监控与历史重测。
            fields += ("monitor", "retests")  # 与 CLI 的公开输出契约一致。
        if not same(read_json(run_dir / "summary.json"), [{field: item[field] for field in fields} for item in replayed]):  # 摘要必须能从已验证的轮次提取。
            raise ValueError("summary differs from replay")  # 不报告篡改后的汇总为正确。
    return {"status": "passed", "evidence_mode": "recorded_replay", "source_evidence_mode": source, "rounds": len(records), "model_calls": 0}  # 历史接口记录回放不产生新的付费调用。


if __name__ == "__main__":  # 支持 python -m bicolor.replay。
    parser = argparse.ArgumentParser(description="不调用 API，复算保存运行的双方奖励")  # 说明验证目的。
    parser.add_argument("--run", type=Path, required=True)  # 指定原始运行目录。
    print(json.dumps(verify_run(parser.parse_args().run), ensure_ascii=False))  # 不一致时返回非零退出码。
