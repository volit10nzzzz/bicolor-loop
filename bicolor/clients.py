import json  # 序列化 API 请求、结构化回答和调用记录。
from threading import Lock  # 在并发出题与求解之间同步额度和日志写入。
from urllib.request import Request, urlopen  # 用标准库调用兼容 Chat Completions 的 HTTP 接口。
from .graph import flip, satisfied  # 离线替身同样使用真实环境规则。


SOLVER_PROMPT = '你是 Solver。翻转一个节点使所有边两端异色，在预算内尽量达到满分。允许暂时掉分。返回 JSON 对象 {"node": 整数节点编号}，每次只选一个节点。'  # 定义单步模型动作。
CHALLENGER_PROMPT = '你是 Challenger。分析上一轮同题多轨迹和题型统计，提出可证伪的行为假设并选择下一轮任务。bridge 必须先掉分；chain 可先持平；size 必须等于输入 fixed_size，不能改变本轮难度组。奖励来自下一轮目标题有效回合的失败比例，不奖励格式错误或新增规模。返回 JSON，包含 family、size、hypothesis、prediction、evidence_episode_ids。prediction 只能为 undo_after_drop、budget_exhausted、solved。引用上一轮真实 episode_id。下一轮将生成两道目标题和一道相反题型的配对对照。'  # 让出题依据来自实际轨迹。
CHALLENGER_PROMPT_V3 = '你是 Challenger。根据输入快照中的轨迹与已结算反馈，为 planned_round 提出可检验的行为假设。feedback.challenger_reward 为空表示此前尚无已结算提案。结合目标与对照的样本数、预测命中和覆盖情况选题。challenge_reward 是有效目标题的预算耗尽比例；对于 undo_after_drop，diagnostic_quality 是目标与对照的命中率之差。两项分开衡量。选 bridge（所有首步掉分）或 chain（可先持平），size 等于 fixed_size。返回 JSON 对象，包含 family、size、hypothesis、prediction、evidence_episode_ids。prediction 只能为 undo_after_drop、budget_exhausted、solved。引用本次输入 episodes 中的真实 episode_id。'  # 预取时以传入快照和计划轮为准。


class ChatClient:  # 两个角色共享调用预算和 HTTP 适配器。
    evidence_mode = "api"  # 记录数据来源，避免混淆离线结果。

    def __init__(self, base_url, api_key, solver_model, challenger_model, log_path, max_calls=64, max_completion_tokens=1024, timeout_seconds=60):  # 配置连接和有限调用预算。
        self.url = base_url.rstrip("/") + "/chat/completions"  # 基地址应终止在 /v1。
        self.api_key = api_key  # 密钥仅保存在内存中，不进入日志。
        self.models = {"solver": solver_model, "challenger": challenger_model}  # 两个角色可以采用不同模型。
        self.log_path = log_path  # 调用记录由实验目录隔离。
        self.max_calls = max_calls  # 约束整次运行的请求总量。
        self.max_completion_tokens = max_completion_tokens  # 约束单次输出长度。
        self.timeout_seconds = timeout_seconds  # 为外部 HTTP 请求设置截止时间。
        self.calls = 0  # 失败请求也占用一次调用预算。
        self.lock = Lock()  # 仅短暂锁住记账或写日志，网络请求仍可并行。

    def ask(self, role, payload):  # 给角色发送结构化上下文并返回 JSON 对象。
        prompt = SOLVER_PROMPT if role == "solver" else CHALLENGER_PROMPT  # 使用对应角色指令。
        if role == "challenger" and "feedback" in payload:  # 新调度提供已完成轮次的反馈。
            prompt = CHALLENGER_PROMPT_V3  # 历史版本继续使用原来的系统消息。
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]  # 发送可审计的完整上下文。
        body = {"model": self.models[role], "messages": messages, "max_completion_tokens": self.max_completion_tokens}  # 保持跨模型调用所需的最少字段。
        request = Request(self.url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})  # 密钥仅进入授权头。
        with self.lock:  # 发送前原子占用一个额度并冻结发送编号。
            if self.calls >= self.max_calls:  # 两个角色使用同一个剩余额度。
                raise RuntimeError("API call budget exhausted")  # 被拒绝的请求不会发往远端。
            self.calls += 1  # 网络或解析失败仍计入开销。
            call = self.calls  # 保持本请求编号独立于其他线程的后续计数。
        record = {"call": call, "request_id": f"request-{call}", "role": role, "request": body}  # 每次尝试有运行内唯一标识，业务身份保存在请求体。
        try:  # 保存外部失败，便于定位模型与网络问题。
            with urlopen(request, timeout=self.timeout_seconds) as response:  # 使用标准 HTTPS 证书校验。
                result = json.load(response)  # 解析实际 HTTP 响应。
            record["response"] = result  # 保留模型原始返回与使用量。
            choice = result["choices"][0]  # 本版只采样一条 completion。
            if choice.get("finish_reason") == "length":  # 被截断的 JSON 不能视为有效决策。
                raise ValueError("model output reached its token limit")  # 请求使用者检查输出预算。
            answer = json.loads(choice["message"]["content"])  # 只接受提示约定的 JSON。
            if not isinstance(answer, dict):  # 动作与提案都必须是对象。
                raise ValueError("model response must be a JSON object")  # 避免将错误结构当作有效动作。
            record["status"] = "ok"  # 标明此调用完成并成功解析。
            return answer  # 返回模型真实输出。
        except Exception as error:  # 为网络、协议和模型输出错误留下记录。
            record["status"] = "error"  # 与成功响应明确区分。
            record["error_type"] = type(error).__name__  # 不记录可能包含凭据的异常对象全文。
            raise  # 失败继续向上层传播，不生成替代答案。
        finally:  # 无论解析或网络是否成功都保存调用记录。
            with self.lock, self.log_path.open("a", encoding="utf-8") as stream:  # 串行落盘整行，允许记录按完成次序出现。
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")  # 一行保存一次完整请求。


class OfflineClient:  # 明确标注的确定性替身，用于零费用运行真实链路。
    evidence_mode = "offline_simulation"  # 离线演示不能当作模型实验结果。
    models = {"solver": "offline-greedy-v1", "challenger": "offline-rules-v3"}  # 记录实际使用的离线策略及版本。

    def ask(self, role, payload):  # 接口与真实 API 一致。
        if role == "solver":  # 模拟一个偏好即时分数的求解器。
            colors = tuple(payload["state"]["colors"])  # 只读取 Solver 可观察状态。
            edges = payload["task"]["edges"]  # 读取公开图结构。
            def priority(node):  # 给每个合法动作计算即时启发式排序。
                score = satisfied(edges, flip(colors, node))  # 优先选择下一状态得分高的动作。
                degree = sum(node in edge for edge in edges)  # 分数相同时偏好连接多的点。
                return score, degree, -node  # 最后按节点编号确定性破同分。
            return {"node": max(range(len(colors)), key=priority)}  # 选择动作，不读取隐藏最短解。
        episodes = payload["episodes"]  # Challenger 消费上一轮真实环境轨迹。
        reversals = [episode for episode in episodes if episode["undo_after_drop"] > 0]  # 找到掉分后立即撤销的样例。
        failures = [episode for episode in episodes if episode["status"] == "budget_exhausted"]  # 收集预算内未解决的样例。
        evidence = reversals or failures or episodes  # 按实际观察选择诊断依据。
        if reversals:  # 观察到撤销行为时保持相同规模复测。
            hypothesis, prediction, family = "Solver 可能在必要掉分后立即撤销。", "undo_after_drop", "bridge"  # 将假设与可计算预测绑定。
        elif failures:  # 没有撤销但耗尽预算时采用较简单的持平路径。
            hypothesis, prediction, family = "Solver 可能难以在预算内完成连续规划。", "budget_exhausted", "chain"  # 记录待检验解释。
        else:  # 成功时在相同规模检验另一类结构。
            hypothesis, prediction, family = "Solver 可能可以解决相同规模的 bridge 题目。", "solved", "bridge"  # 明确这是下一轮预测。
        feedback = payload.get("feedback", {})  # 旧版观察或种子快照允许尚无结算奖励。
        evaluation = feedback.get("challenger_reward")  # 只使用决策前已经可见的评价。
        if evaluation and evaluation["diagnostic_quality"] is not None and evaluation["diagnostic_quality"] <= 0:  # 先前选的结构没有更频繁地复现撤销。
            family = "chain" if feedback["proposal"]["family"] == "bridge" else "bridge"  # 换到配对题型再检验同一事件。
            hypothesis, prediction = f"此前撤销命中率没有高于对照组，改用 {family} 检验掉分后撤销。", "undo_after_drop"  # 保持预测事件可比。
        size = payload["fixed_size"]  # 冻结难度组，出题策略只能选择题型。
        return {"family": family, "size": size, "hypothesis": hypothesis, "prediction": prediction, "evidence_episode_ids": [item["episode_id"] for item in evidence]}  # 返回有证据引用的出题条件。
