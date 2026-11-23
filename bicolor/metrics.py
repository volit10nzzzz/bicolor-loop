from collections import Counter, defaultdict  # 按计划题、状态及证书分层汇总。


def _outcomes(episodes):  # 统一成功率的有效分母及中断计数。
    statuses = Counter(item["status"] for item in episodes)  # 每条记录只占一个状态。
    solved = statuses["solved"]  # 真实达成满分的终局数。
    valid = solved + statuses["budget_exhausted"]  # 正常成功和预算耗尽均是有效结果。
    return {  # 同时返回原始数，使比例可以从日志直接复算。
        "solved": solved, "budget_exhausted": statuses["budget_exhausted"],  # 正常终局。
        "invalid_action": statuses["invalid_action"], "interrupted": statuses["interrupted"],  # 模型非法节点与请求故障分别计数。
        "completed": valid + statuses["invalid_action"], "valid": valid,  # 无效动作结束回合，中断没有终局。
        "success_rate": solved / valid if valid else None,  # 无有效样本时比例未知。
    }  # 返回本分层的实际样本覆盖。


def _prediction(event, members, tasks, attempts):  # 在一个决策的一个来源组内评价预注册事件。
    expected = {task["task_id"]: attempts for task in tasks}  # 每道题都必须具备全部计划作答。
    actual = Counter(item["task_id"] for item in members)  # 汇总实际回合，防止同组多题数量相互抵消。
    valid = [item for item in members if item["status"] in ("solved", "budget_exhausted")]  # 事件率只使用有效终局。
    hits = sum(item["undo_after_drop"] > 0 for item in valid) if event == "undo_after_drop" else sum(item["status"] == event for item in valid)  # 一条轨迹最多命中一次。
    complete = bool(expected) and actual == expected and len(valid) == len(members)  # 缺题、无效动作或中断均使本组评价未知。
    return {"hits": hits, "episodes": len(members), "planned": sum(expected.values()), "valid": len(valid), "complete": complete}  # 命中数和有效数分别报告。


def _retest_counts(episodes):  # 历史同题按实际回合聚合两种事件。
    outcomes = _outcomes(episodes)  # 与分层监控采用相同终局口径。
    hits = sum(item["undo_after_drop"] > 0 for item in episodes if item["status"] in ("solved", "budget_exhausted"))  # 按发生过撤销的有效回合数计。
    return dict(outcomes, episodes=len(episodes), undo_hits=hits, undo_rate=hits / outcomes["valid"] if outcomes["valid"] else None)  # 保存分子分母。


def _retests(tasks, indexed, previous_records):  # 找到每道重测题最早的原始目标样本。
    current = defaultdict(list)  # 原题标识跨轮稳定，题面编号在重测中可以变化。
    for task in tasks:  # 以计划题为起点，因此未完成的历史题仍可报告。
        if task["group"] == "historical_retest":  # 只比较额外调度的历史组。
            current[task["origin_task_id"]].extend(indexed[task["task_id"]])  # 同一原题的所有实际回合放在一起。
    if not current:  # 默认关闭重测时无需读历史轨迹。
        return []  # 无重测计划。
    original = {}  # 每个原题只记录首次评测。
    for record in previous_records:  # 调度按轮次顺序保存已完成记录。
        by_origin = defaultdict(list)  # 当前历史轮内聚合同题多条作答。
        for episode in record["episodes"]:  # 直接读取首次实际轨迹以便独立重算。
            origin = episode["origin_task_id"]  # 不根据家族或结构猜测原题关联。
            if episode["group"] == "target" and origin in current and origin not in original:  # 对照和历史组都不能充当首次作答。
                by_origin[origin].append(episode)  # 同轮同原题多条一并归集。
        for origin, episodes in by_origin.items():  # 固定本轮首次出现的原题成绩。
            original[origin] = record["round"], _retest_counts(episodes)  # 保留原始观察轮次。
    result = []  # 一条结果对应一个原题。
    for origin, episodes in current.items():  # 按实际重测计划的顺序生成。
        original_round, before = original[origin]  # 调度只允许选此前已求解的真实目标题。
        after = _retest_counts(episodes)  # 当前结果可能只有中断或没有实际回合。
        row = {"origin_task_id": origin, "original_round": original_round, "original": before, "retest": after}  # 便于还原所有计数。
        for key in ("success_rate", "undo_rate"):  # 两个变化量都按有效回合率比较。
            row[key + "_change"] = after[key] - before[key] if before[key] is not None and after[key] is not None else None  # 仅在双方有分母时报告差。
        result.append(row)  # 每个原题只输出一次。
    return result  # 返回重测统计。


def measure(proposal, episodes, decision, round_index, tasks, rollouts_per_task, solver_model, previous_records):  # 从已知计划、轨迹与实际动作计算整轮评价。
    from .loop import summarize  # 延迟导入调度模块，复用旧版按题等权得分。
    decision_id = decision["decision_id"] if decision is not None else None  # 种子也分组观察，但不发出题奖励。
    predictions = {}  # 保存目标和对照各自的预注册事件统计。
    for group in ("target", "control"):  # 历史题即使同题型也不会进入此处。
        chosen = [item for item in episodes if item["group"] == group and item["decision_id"] == decision_id]  # 显式筛选决策和来源。
        scheduled = [task for task in tasks if task["group"] == group and task["decision_id"] == decision_id]  # 按实际题单检查样本是否齐全。
        predictions[group] = _prediction(proposal["prediction"], chosen, scheduled, rollouts_per_task)  # 组内统计命中。
    target, control = predictions["target"], predictions["control"]  # 目标评价与对照评价可分别缺失。
    evaluation = None  # 种子轮没有可归属的出题动作。
    if decision is not None:  # 只有真实提案才结算奖励。
        targets = [item for item in episodes if item["group"] == "target" and item["decision_id"] == decision_id]  # 保存挑战分的证据列表。
        failures = sum(item["status"] == "budget_exhausted" for item in targets)  # 只计预算内失败。
        challenge = failures / target["valid"] if target["complete"] else None  # 挑战分不依赖对照组。
        quality = None  # 证据不足或终局预测均无数值质量分。
        if proposal["prediction"] != "undo_after_drop":  # 终局事件与挑战奖励信息重复。
            quality_status = "not_applicable"  # 区别于采样异常。
        elif target["complete"] and control["complete"]:  # 两组均覆盖计划且全部为有效终局。
            quality = target["hits"] / target["valid"] - control["hits"] / control["valid"]  # 预注册行为的命中率差。
            quality_status = "scored"  # 两个分母都有效。
        else:  # 可能只有一个分组不完整。
            quality_status = "inconclusive"  # 保留未知质量。
        evaluation = {  # 明确保留兼容字段，并记录实际决策归属。
            "decision_id": decision_id, "source_round": decision["observation_round"], "observation_round": decision["observation_round"],  # 两个观察轮次字段同义。
            "planned_round": decision["planned_round"], "evaluation_round": round_index,  # 不从实际评价轮次反推观察轮次。
            "status": "scored" if challenge is not None else "inconclusive", "reward": challenge, "challenge_reward": challenge,  # 标量只代表目标挑战分。
            "failures": failures, "target_episodes": target["episodes"], "episode_ids": [item["episode_id"] for item in targets],  # 保存原有可审计样本。
            "diagnostic_quality": quality, "quality_status": quality_status,  # 质量分与挑战分保持独立。
        }  # 上述字段都属于被评价的出题动作。
    indexed = defaultdict(list)  # 所有计划题都有独立统计，即使尚无人作答。
    for item in episodes:  # 将实际回合关联到本轮执行题，不混用历史原题号。
        indexed[item["task_id"]].append(item)  # 记录完整和异常回合。
    strata = defaultdict(list)  # 按题型、规模、最短距离、首步条件、样本来源分层。
    families = defaultdict(list)  # 同时为旧格式的按题型摘要提供计划和实际分母。
    for task in tasks:  # 未执行的计划题也进入覆盖监控。
        cert = task["certificate"]  # 使用环境核验过的结构证书。
        key = task["family"], task["size"], cert["min_flips"], cert["all_first_moves_drop"], task["group"]  # 明确结构与求解结果的含义。
        strata[key].append(task["task_id"])  # 同一层可能包含多道原题。
        families[task["family"]].append(task["task_id"])  # 后续按题型汇总。
    monitor = []  # 依排程顺序输出分层结果，确保可回放。
    for (family, size, distance, first_drop, group), ids in strata.items():  # 每一层按真实执行模型报告。
        outcomes = _outcomes([item for key in ids for item in indexed[key]])  # 成功率分母与计划量分开计算。
        monitor.append(dict(outcomes, solver_model=solver_model, family=family, size=size, min_flips=distance, all_first_moves_drop=first_drop, group=group, planned=len(ids) * rollouts_per_task))  # 样本来源独立分层。
    finished = [item for ids in families.values() for key in ids for item in indexed[key] if item["status"] != "interrupted"]  # 部分轨迹不可作为最终得分。
    summary = summarize(finished)  # 保留旧版按题等权的均分和观察分层。
    for family, ids in families.items():  # 即使同一题型全未完成，仍保留其计划量。
        row = summary.setdefault(family, {"mean_score": None, "stratum": None, "tasks": 0, "episodes": 0, "invalid_actions": 0, "undo_after_drop": 0})  # 缺少得分就保留空值。
        row.update(_outcomes([item for key in ids for item in indexed[key]]), planned=len(ids) * rollouts_per_task)  # 旧摘要也显示真实成功分母。
    coverage = {"structure_key": f"{proposal['family']}:{proposal['size']}", "new_target_structure": not any(record["proposal"]["family"] == proposal["family"] and record["proposal"]["size"] == proposal["size"] for record in previous_records)}  # 同构重编号不增加目标题型覆盖。
    return {"summary": summary, "prediction_results": predictions, "challenger_reward": evaluation, "coverage": coverage, "monitor": monitor, "retests": _retests(tasks, indexed, previous_records)}  # 原始输入足以重建每项指标。
