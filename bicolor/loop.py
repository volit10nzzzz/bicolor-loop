from dataclasses import asdict  # 将不可变任务写入可复现记录。
import math  # 校验奖励参数是有限数。
import json  # 保存轮次、轨迹和角色输入。
from statistics import mean  # 按题和题型计算均分。
from .graph import audit, flip, generate_task, satisfied  # 共用真实环境和验证器。


def solver_observation(episode_id, public_task, colors, score, remaining_steps, history, completion_bonus):  # 在线采样与训练导出共用确切观察格式。
    return {  # 环境只向策略传递此时可见的信息。
        "episode_id": episode_id,  # 关联本次回合。
        "task": public_task,  # 公开图和原始预算。
        "state": {"colors": colors, "score": score, "remaining_steps": remaining_steps},  # 当前着色与剩余步数。
        "history": history,  # 当前动作之前的交互。
        "reward_contract": {"completion_bonus": completion_bonus, "gamma": 1.0},  # 约定不折扣目标。
    }  # 结束观察。


def solve(task, client, episode_id, on_interrupt=None, completion_bonus=1.0, keep_action=False, before_step=None):  # 执行求解，允许运行器保存异常中断的部分轨迹。
    colors = task.colors  # 每次 rollout 都重置初始状态。
    steps = []  # 保存动作前后状态和精确评分。
    status = "budget_exhausted"  # 用完循环仍未成功时保留此终止原因。
    for index in range(task.budget):  # 步数预算是真实环境上限。
        before = satisfied(task.edges, colors)  # 获取动作前整数分数。
        payload = solver_observation(episode_id, task.public(), colors, before / len(task.edges), task.budget - index, steps, completion_bonus)  # 构建当前动作的可见输入。
        try:  # 外部请求可能在已执行若干步后失败。
            if before_step is not None:  # 后台错误也可能中断已执行若干步的回合。
                before_step()  # 先保存中断证据，再由调度器向上传播异常。
            action = client.ask("solver", payload)  # API 或显式离线替身给出下一步动作。
        except Exception as error:  # 保存中断证据后继续传播失败。
            if on_interrupt is not None:  # 由运行器负责持久化，不在环境层选择路径。
                on_interrupt({"episode_id": episode_id, "task_id": task.task_id, "family": task.family, "size": task.size, "task": task.public(), "steps": steps, "status": "interrupted", "error_type": type(error).__name__, "final_score": before / len(task.edges)})  # 部分轨迹不能标成预算终局。
            raise  # 不用离线动作代替失败请求。
        node = action.get("node")  # 读取约定的节点编号。
        try:  # 只有外部动作边界需要验证节点。
            after_colors = flip(colors, node)  # 执行一次合法翻色。
        except ValueError:  # 无效动作消耗一次尝试并结束当前回合。
            status = "invalid_action"  # 单独记录格式或节点错误。
            steps.append({"valid": False, "action": action, "before": colors, "after": colors, "delta_reward": 0.0, "completion_bonus": 0.0, "reward": 0.0, "delta_edges": 0})  # 无效动作不伪装成有效状态转移。
            if keep_action:  # 无效动作也保留原始回答。
                steps[-1]["raw_action"] = action  # 供按步核验。
            break  # 保持失败可见，不代替模型选择节点。
        after = satisfied(task.edges, after_colors)  # 验证器直接计算动作后得分。
        delta = (after - before) / len(task.edges)  # 真实边数分差归一化到题目规模。
        bonus = completion_bonus if after == len(task.edges) else 0.0  # 满分时立即结束，只奖励一次。
        steps.append({  # 保存模型采取的环境动作与奖励分解。
            "valid": True, "node": node,  # 有效翻转节点。
            "before": colors, "after": after_colors,  # 环境转移前后着色。
            "satisfied_before": before, "satisfied_after": after,  # 整数计分便于核对。
            "delta_edges": after - before, "delta_reward": delta,  # 当步约束改善。
            "completion_bonus": bonus, "reward": delta + bonus,  # 成功加成与总收益。
            "score": after / len(task.edges), "remaining_steps_before": task.budget - index,  # 还原有限时域观察。
        })  # 到此完成一次状态转移。
        if keep_action:  # 新版同时保留环境动作与完整模型回答。
            steps[-1]["raw_action"] = action  # 额外说明字段可随回放核对。
        colors = after_colors  # 推进环境状态。
        if after == len(task.edges):  # 所有边异色即成功。
            status = "solved"  # 标记真实终局。
            break  # 达成目标后不继续消耗 API。
    reversals = sum(previous["valid"] and current["valid"] and previous["delta_edges"] < 0 and previous["node"] == current["node"] for previous, current in zip(steps, steps[1:]))  # 机械识别掉分后立即撤销。
    return {"episode_id": episode_id, "task_id": task.task_id, "family": task.family, "size": task.size, "task": task.public(), "steps": steps, "status": status, "final_score": satisfied(task.edges, colors) / len(task.edges), "undo_after_drop": reversals}  # 返回完整诊断输入。


def summarize(episodes):  # 先计算单题均分，再计算题型均分。
    summary = {}  # 保存当前轮实际出现的题型。
    for family in ("bridge", "chain"):  # 当前生成器支持两个题型。
        members = [episode for episode in episodes if episode["family"] == family]  # 只统计对应题型。
        if not members:  # 未采样题型没有可报告分数。
            continue  # 不把缺少数据当作零分。
        task_ids = sorted({episode["task_id"] for episode in members})  # 同题多轨迹先归组。
        task_means = [mean(item["final_score"] for item in members if item["task_id"] == task_id) for task_id in task_ids]  # 每道题内部求均分。
        score = mean(task_means)  # 各道题等权进入题型均分。
        stratum = "low" if score < 0.6 else "middle" if score < 0.9 else "high"  # 演示分层阈值，供 Challenger 参考。
        summary[family] = {"mean_score": score, "stratum": stratum, "tasks": len(task_ids), "episodes": len(members), "success_rate": mean(item["status"] == "solved" for item in members), "invalid_actions": sum(item["status"] == "invalid_action" for item in members), "undo_after_drop": sum(item["undo_after_drop"] for item in members)}  # 保留样本量和行为信息。
    return summary  # 返回可审计的题型统计。


def validate_proposal(proposal, episodes, fixed_size=None):  # 模型提案只允许进入已验证的任务空间。
    if proposal.get("family") not in ("bridge", "chain") or type(proposal.get("size")) is not int or not 2 <= proposal["size"] <= 4:  # 校验生成参数。
        raise ValueError("Challenger must select family bridge/chain and integer size 2..4")  # 错误提案明确终止运行。
    if not isinstance(proposal.get("hypothesis"), str) or not proposal["hypothesis"].strip():  # 每次出题必须说明检验目的。
        raise ValueError("Challenger must provide a hypothesis")  # 禁止无诊断内容的成功记录。
    if proposal.get("prediction") not in ("undo_after_drop", "budget_exhausted", "solved"):  # 只接受可从轨迹计算的预测。
        raise ValueError("unsupported Challenger prediction")  # 避免无法验证的自由文本标签。
    if fixed_size is not None and proposal["size"] != fixed_size:  # 同一次实验不允许通过增加规模获得更高失败率。
        raise ValueError("Challenger must keep the fixed task size")  # 明确拒绝改变难度组。
    references = proposal.get("evidence_episode_ids")  # 取得提案引用的上一轮证据。
    known = {episode["episode_id"] for episode in episodes}  # 建立实际存在的轨迹集合。
    if not isinstance(references, list) or not references or any(not isinstance(item, str) or item not in known for item in references):  # 引用必须非空且全部存在。
        raise ValueError("Challenger evidence must cite existing previous-round episodes")  # 拒绝模型编造证据 ID。


def prediction_results(proposal, episodes):  # 统计预先声明的行为在目标题和对照题上的发生次数。
    result = {}  # 分组报告，避免仅报告更有利的一组。
    for group in ("target", "control"):  # 两类题共享规模、初始分和最短步数。
        chosen = [episode for episode in episodes if (episode["family"] == proposal["family"]) == (group == "target")]  # 根据目标题型归组。
        if proposal["prediction"] == "undo_after_drop":  # 掉分撤销采用明确的事件计数。
            hits = sum(episode["undo_after_drop"] > 0 for episode in chosen)  # 每条轨迹最多计一次命中。
        else:  # 其余预测可以直接比较终止状态。
            hits = sum(episode["status"] == proposal["prediction"] for episode in chosen)  # 检查是否如预测成功或耗尽预算。
        result[group] = {"hits": hits, "episodes": len(chosen)}  # 保留分母，不把次数当概率。
    return result  # 观察支持程度由读者结合样本量判断。


def run_rounds(client, output, seed=42, rounds=2, rollouts_per_task=2, initial_size=2, completion_bonus=1.0):  # 串起完整的冻结模型外循环。
    if rounds < 2 or rollouts_per_task < 1:  # 完整示例至少需要两轮和每题一次求解。
        raise ValueError("use at least two rounds and one rollout per task")  # 防止把单轮误报为外循环完成。
    if not math.isfinite(completion_bonus) or completion_bonus < 0:  # 奖励配置不能产生非有限值或反向完成激励。
        raise ValueError("completion_bonus must be finite and nonnegative")  # 在任何模型调用前失败。
    output.mkdir(parents=True, exist_ok=True)  # CLI 或测试为本次运行提供独立目录。
    records = []  # 保存已完成轮次，供最终汇总。
    proposal = {"family": "bridge", "size": initial_size, "hypothesis": "种子轮观察必要掉分与持平路径上的行为。", "prediction": "undo_after_drop", "evidence_episode_ids": []}  # 初轮没有历史证据。
    with (output / "rounds.jsonl").open("x", encoding="utf-8") as rounds_file, (output / "episodes.jsonl").open("x", encoding="utf-8") as episodes_file:  # 拒绝覆盖已有实验。
        def save_episode(episode):  # 完整和中断轨迹使用同一落盘路径。
            episodes_file.write(json.dumps(episode, ensure_ascii=False) + "\n")  # 保存原始回合状态。
            episodes_file.flush()  # 立即保留已完成工作。
        for round_index in range(rounds):  # 外循环按轮推进。
            opposite = "chain" if proposal["family"] == "bridge" else "bridge"  # 生成结构条件相反的对照。
            families = [proposal["family"], proposal["family"], opposite]  # 分配两道目标题与一道对照题。
            tasks = [generate_task(family, proposal["size"], seed + round_index * 100 + index, f"r{round_index}-q{index}") for index, family in enumerate(families)]  # 每轮使用新种子并可复现。
            (output / f"schedule-{round_index}.json").write_text(json.dumps({"round": round_index, "proposal": proposal, "tasks": [asdict(task) for task in tasks]}, ensure_ascii=False, indent=2), encoding="utf-8")  # 求解前保留调度输入，支持中断审计。
            episodes = []  # 同一轮完整轨迹在此聚合。
            for task in tasks:  # 同题多采样前保持题目固定。
                for attempt in range(rollouts_per_task):  # API 模型可能在同题上产生不同路径。
                    episode = solve(task, client, f"{task.task_id}-e{attempt}", save_episode, completion_bonus)  # 传入保存器与奖励配置。
                    episodes.append(episode)  # 为 Challenger 收集完整轨迹。
                    save_episode(episode)  # 每完成一条就保留，故障时仍可审计。
            record = {"round": round_index, "source_round": round_index - 1 if round_index else None, "evidence_mode": client.evidence_mode, "proposal": proposal, "tasks": [dict(asdict(task), certificate=audit(task)) for task in tasks], "episodes": episodes, "summary": summarize(episodes), "prediction_results": prediction_results(proposal, episodes), "next_proposal": None}  # 把提案、任务、结果绑定为同一轮证据。
            record["challenger_reward"] = evaluate_challenger(proposal, episodes, round_index) if round_index else None  # 种子轮不归功于出题模型。
            record["coverage"] = {"structure_key": f"{proposal['family']}:{proposal['size']}", "new_target_structure": not any(item["proposal"]["family"] == proposal["family"] and item["proposal"]["size"] == proposal["size"] for item in records)}  # 按结构组记录覆盖，重命名不算新增。
            try:  # 出题失败也保留已经完成的轮次。
                if round_index + 1 < rounds:  # 只有后面还有轮次才调用 Challenger。
                    challenger_input = {"round": round_index, "summary": record["summary"], "episodes": episodes, "previous_proposal": proposal, "fixed_size": initial_size}  # 下一轮选择直接依赖本轮轨迹。
                    next_proposal = client.ask("challenger", challenger_input)  # 通过同一角色接口获得真实或模拟提案。
                    validate_proposal(next_proposal, episodes, initial_size)  # 验证结构和证据引用。
                    decision = {"decision_id": f"decision-{round_index}", "source_round": round_index, "input": challenger_input, "output": next_proposal}  # 保存出题前后的可追溯关系。
                    (output / f"decision-{round_index}.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")  # 单独提供便于人工审阅的出题记录。
                    record["next_proposal"] = next_proposal  # 内存汇总包含下一轮决策。
                    proposal = next_proposal  # 实际使用 Challenger 决策推进循环。
            finally:  # 保存实际提案或尚未获得提案的状态。
                rounds_file.write(json.dumps(record, ensure_ascii=False) + "\n")  # 与内存中的轮次证据保持一致。
                rounds_file.flush()  # 及时落盘。
            records.append(record)  # 汇总已经完成的轮次。
    return records  # 返回带完整决策链的结果。


def evaluate_challenger(proposal, episodes, evaluation_round):  # 为上一轮提案计算延迟奖励。
    targets = [item for item in episodes if item["family"] == proposal["family"]]  # 对照组只作诊断，不参与奖励。
    eligible = bool(targets) and all(item["status"] in ("solved", "budget_exhausted") for item in targets)  # 不用无效动作或运行故障替出题方获奖。
    failures = sum(item["status"] == "budget_exhausted" for item in targets)  # 仅统计有效回合的预算失败。
    return {"decision_id": f"decision-{evaluation_round - 1}", "source_round": evaluation_round - 1, "evaluation_round": evaluation_round, "status": "scored" if eligible else "inconclusive", "reward": failures / len(targets) if eligible else None, "failures": failures, "target_episodes": len(targets), "episode_ids": [item["episode_id"] for item in targets]}  # 未知奖励保持空值，不能当成零。
