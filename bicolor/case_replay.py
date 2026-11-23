"""用录制动作重建基线、提案评价和下一题的真实交接。"""  # 此入口全程离线。
import argparse  # 指定公开的案例文件。
import json  # 读取案例及比较原始记录。
from pathlib import Path  # 接受本地文件路径。
from tempfile import TemporaryDirectory  # 隔离调度器的决策文件。
from dataclasses import asdict  # 构造与正式批次一致的题单字段。
from .curriculum import prepare  # 复用生产调度器的观察构造和新题生成。
from .graph import audit, flip, generate_task, satisfied  # 按基线种子重建图与证书。
from .loop import solve, solver_observation  # 重算动作、奖励与结束原因。
from .metrics import measure  # 按正式口径汇总当轮结果。


def baseline(client):  # 两题沿用原实验身份、种子和五步预算。
    episodes, tasks = [], []  # 区分已完成轨迹与事先确定的题单。
    proposal = {"family": "bridge", "size": 4, "prediction": "undo_after_drop", "hypothesis": "基线配对观察", "evidence_episode_ids": []}  # 基线组别由实验者指定。
    for family, seed, group in (("bridge", 42, "target"), ("chain", 44, "control")):  # 沿用原图。
        task = generate_task(family, 4, seed, family)  # 编号与旧求解观察完全一致。
        ep = solve(task, client, family + "-0", keep_action=True)  # 每步重新构造观察。
        ep.update(group=group, origin_task_id=family, decision_id=None)  # 添加观察之后才使用的归属标签。
        episodes.append(ep)  # 保存基线证据。
        tasks.append(dict(asdict(task), group=group, origin_task_id=family, decision_id=None, certificate=audit(task)))  # 加入题面证书。
    metrics = measure(proposal, episodes, None, 0, tasks, 1, client.models["solver"], [])  # 种子没有模型提案奖励。
    return dict(round=0, decision_id=None, proposal=proposal, episodes=episodes, tasks=tasks, **metrics)  # 作为下一次提案的完整快照。


def probe(client, output, first):  # 只运行预先约定的一道目标题。
    proposal, decision, items = prepare(client, output, (first,), 1, 42, 4, 0, client.models)  # 先保存提案再生成新题。
    task = items[0][0]  # 固定取第一道目标，无论模型选择哪一类。
    episode = solve(task, client, task.task_id + "-e0", keep_action=True)  # 新 Solver 只收本题当前观察。
    return decision, episode  # 单题验收，未运行配对对照。


def settle_probe(client, first, decision, episode):  # 用已完成的目标回合构造下一次决策的冻结反馈。
    proposal = decision["output"]  # 只按新题之前登记的预测结算。
    task = generate_task(proposal["family"], 4, 142, episode["task_id"])  # 第一批新题的首道目标种子。
    return grade_target(client, task, decision, episode, [first])  # 调用与第二份提案相同的结算代码。


def grade_target(client, task, decision, episode, records):  # 将一次目标题作答归属给对应的出题决策。
    number = decision["planned_round"]  # 两轮案例均按预定轮次实际执行。
    group = dict(group="target", origin_task_id=task.task_id, decision_id=decision["decision_id"])  # 单题没有配对对照。
    task_row = dict(asdict(task), certificate=audit(task), **group)  # 原始题面和可解性证书。
    member = dict(episode, **group)  # 将求解回合关联到同一提案。
    result = measure(decision["output"], [member], decision, number, [task_row], 1, client.models["solver"], records)  # 按正式评价口径重算。
    return dict(round=number, decision_id=decision["decision_id"], proposal=decision["output"], episodes=[member], tasks=[task_row], **result)  # 保留真正出现的目标题观察。


def handoff(client, output, first, latest):  # 让第二次挑战提案落实到一个真实求解请求。
    _, decision, items = prepare(client, output, (first, latest), 2, 42, 4, 0, client.models)  # 传入已结算轨迹，固定规模。
    task = items[0][0]  # 预先固定取首道目标，与上次实验相同。
    score = satisfied(task.edges, task.colors) / len(task.edges)  # 验证器计算初始得分。
    observation = solver_observation(task.task_id + "-e0", task.public(), task.colors, score, task.budget, [], 1.0)  # 新回合从空历史开始。
    action = client.ask("solver", observation)  # 一次真实回复即可核实题目已送达。
    return decision, task, action  # 保留首步回答，不产生终局评价。


class Recorded:  # 用已保存的消息替代两个模型。
    def __init__(self, case):  # 关联当时的模型和请求顺序。
        self.models = case["models"]  # 重建相同决策记录。
        self.calls = iter(case["calls"])  # 按本例同步运行次序消费。
    def ask(self, role, payload):  # 比较当步应见的全部信息。
        row = next(self.calls)  # 缺步骤即失败。
        if role != row["role"] or json.dumps(payload, sort_keys=True) != json.dumps(row["input"], sort_keys=True):  # JSON比较也适用于元组和列表。
            raise ValueError("recorded observation mismatch")  # 状态或历史发生偏差时停止。
        return row["output"]  # 执行真实保存的动作。


def finish_probe(client, task, first_call, decision, records):  # 恢复同一题的求解，首步沿用上次确切的模型回答。
    prefix = Recorded({"models": client.models, "calls": [first_call]})  # 只含已实际发出的首步请求和响应。
    class ResumedSolver:  # 重新走原求解循环，保证行为历史、颜色、剩余预算和评分一致。
        def ask(self, role, payload):  # 真实采样与离线回放均使用此边界。
            if not payload["history"]:  # 首个观察必须和交接记录完全一致。
                return prefix.ask(role, payload)  # 复用首步，不再次请求模型。
            return client.ask(role, payload)  # 其余步骤由调用方指定的客户端作答。
    episode = solve(task, ResumedSolver(), task.task_id + "-e0", keep_action=True)  # 保留所有原始动作直到真实终局。
    return episode, grade_target(client, task, decision, episode, records)  # 计算第二份提案预登记事件的命中。


if __name__ == "__main__":  # 提供不需要密钥的公开演示入口。
    parser = argparse.ArgumentParser()  # 接受主案例及可选的续轮交接记录。
    parser.add_argument("case", type=Path)  # 显式指定来源。
    parser.add_argument("--handoff", type=Path)  # 可选的第二次决策与接单记录。
    parser.add_argument("--outcome", type=Path)  # 可选的第二份提案后续动作及完整结算。
    args = parser.parse_args()  # 解析两个案例路径。
    if args.outcome and not args.handoff:  # 后续动作依赖已冻结的交接首步。
        parser.error("--outcome requires --handoff")  # 防止把记录当作独立任务。
    case = json.loads(args.case.read_text())  # 读取公开的脱敏交互。
    client = Recorded(case)  # 禁用网络依赖。
    first = baseline(client)  # 先验算基线。
    with TemporaryDirectory() as directory:  # 不修改原始实验。
        decision, episode = probe(client, Path(directory), first)  # 再验算出题与新题。
    if next(client.calls, None) is not None or json.loads(json.dumps(episode)) != case["new_episode"] or decision["output"] != case["proposal"]:  # 核对终局及最后动作。
        raise ValueError("recorded outcome mismatch")  # 拒绝不一致的完整性声明。
    latest = settle_probe(client, first, decision, episode)  # 将第一次新题的结果归给出题动作。
    print("回放通过", latest["proposal"]["prediction"], episode["status"], "预测命中", latest["prediction_results"]["target"]["hits"], "挑战分", latest["challenger_reward"]["challenge_reward"], "诊断质量", latest["challenger_reward"]["diagnostic_quality"])  # 打印第一次评价。
    if args.handoff:  # 可继续核验第二次真实交接。
        continuation = json.loads(args.handoff.read_text())  # 只读脱敏的原始请求与回答。
        if continuation["models"] != client.models:  # 本案例约定双角色使用同一冻结模型标识。
            raise ValueError("handoff model mismatch")  # 防止用其它模型的交接冒充续轮。
        second = Recorded(continuation)  # 继续使用录制动作。
        with TemporaryDirectory() as directory:  # 隔离第二次提案记录。
            decision, task, action = handoff(second, Path(directory), first, latest)  # 重建全部反馈、题面与首步观察。
        if next(second.calls, None) is not None:  # 案例边界止于首次接单作答。
            raise ValueError("unexpected calls after handoff")  # 不隐去额外请求。
        after = flip(task.colors, action["node"])  # 用真实验证器执行收到的合法首步。
        receipt = {"task_id": task.task_id, "node": action["node"], "colors_after": list(after), "satisfied_after": satisfied(task.edges, after)}  # 仅计算已经发生的转移。
        if receipt != continuation["first_step"]:  # 确认回放能重建原先保存的接单结果。
            raise ValueError("handoff first step mismatch")  # 拒绝错误的节点或图状态。
        print("交接回放通过", decision["decision_id"], decision["output"]["prediction"], task.task_id, action, "首步满足边数", receipt["satisfied_after"])  # 报告真实的新提案与首步。
        if args.outcome:  # 对照完整的第二题终局与出题奖励。
            outcome = json.loads(args.outcome.read_text())  # 读取除首步外的剩余调用。
            tail = Recorded(outcome)  # 回放禁止重新采样。
            if tail.models != second.models:  # 模型标识应与已接题策略一致。
                raise ValueError("outcome model mismatch")  # 防止记录串接到不同模型。
            member, report = finish_probe(tail, task, continuation["calls"][1], decision, [first, latest])  # 前缀来自已经核对的首步。
            if next(tail.calls, None) is not None or json.loads(json.dumps(member)) != outcome["episode"]:  # 验证终局、回报和全部状态。
                raise ValueError("second episode mismatch")  # 拒绝缺步、多步或事后改分。
            if report["challenger_reward"] != outcome["evaluation"] or report["prediction_results"] != outcome["prediction_results"]:  # 独立结算预登记预测。
                raise ValueError("second proposal evaluation mismatch")  # 拒绝虚报预测命中或奖励。
            reward = report["challenger_reward"]  # 各项结果都来自实际决策归属。
            hit = report["prediction_results"]["target"]  # 同一条轨迹多次发生事件也只算一次命中。
            print("第二轮回放通过", member["status"], "Solver累计", sum(step["reward"] for step in member["steps"]), "撤销次数", member["undo_after_drop"], "目标命中", hit["hits"], "/", hit["valid"], "挑战分", reward["challenge_reward"], "诊断分", reward["diagnostic_quality"], "对照样本数", report["prediction_results"]["control"]["valid"])  # 保留原始分母与缺失的诊断量。
