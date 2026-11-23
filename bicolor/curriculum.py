from dataclasses import asdict  # 序列化原题与重测题。
import json  # 保存可核验的轮次与观察。
import math  # 校验奖励配置。
import random  # 独立采样历史原题。
from .scheduling import run_async, run_sync  # 两种交接方式共用同一题目和求解实现。
from .graph import Task, audit, generate_task  # 复用真实构造和可解性验证。
from .loop import solve, validate_proposal  # 与旧版共享环境转移与提案契约。
from .metrics import measure  # 汇总分组结果和延迟奖励。


def write_json(path, value):  # 各批次写独立文件，保留首次调度证据。
    with path.open("x", encoding="utf-8") as stream:  # 禁止覆盖已有证据。
        json.dump(value, stream, ensure_ascii=False, indent=2)  # 便于人工核验。


def append_json(path, value):  # 每个流只有主线程写入。
    with path.open("a", encoding="utf-8") as stream:  # 已完成的回合及时落盘。
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")  # 每行一个完整事件。


def prepare(client, output, snapshot, planned, seed, fixed_size, history_count, models):  # 只消费给定截止时间的已完成轮次。
    decision = None  # 种子批不归属任何模型提案。
    proposal = {"family": "bridge", "size": fixed_size, "prediction": "undo_after_drop", "hypothesis": "观察必要掉分后的撤销。", "evidence_episode_ids": []}  # 初始课程。
    if snapshot:  # 初轮之后通过模型选择题型。
        latest = snapshot[-1]  # 快照只包含已结束的回合和评价。
        identity = f"decision-{planned - 1}"  # 分配唯一动作编号，即使两次读取同一快照。
        feedback = {key: latest[key] for key in ("decision_id", "proposal", "prediction_results", "challenger_reward", "coverage", "monitor", "retests")}  # 显式反馈上次已观察的结果。
        observation = {"decision_id": identity, "planned_round": planned, "round": latest["round"], "fixed_size": fixed_size, "summary": latest["summary"], "episodes": latest["episodes"], "previous_proposal": latest["proposal"], "feedback": feedback}  # 不携带未完成轮次。
        proposal = client.ask("challenger", observation)  # 获取本次出题动作。
        validate_proposal(proposal, latest["episodes"], fixed_size)  # 检查引用和难度组。
        decision = {"decision_id": identity, "observation_round": latest["round"], "source_round": latest["round"], "planned_round": planned, "challenger_model": models["challenger"], "input": observation, "output": proposal}  # 明确观察、动作和角色。
        write_json(output / f"{identity}.json", decision)  # 保留已实际发生的提案。
    target = proposal["family"]  # 本轮目标题型。
    control = "chain" if target == "bridge" else "bridge"  # 配对的另一种题型。
    items = []  # 保存任务和分组元信息。
    for index, family in enumerate((target, target, control)):  # 保持原有配比。
        task = generate_task(family, fixed_size, seed + planned * 100 + index, f"r{planned}-q{index}")  # 新题种子独立于历史抽样。
        items.append((task, "target" if index < 2 else "control", task.task_id))  # 记录原始题号。
    if history_count == 0:  # 开关关闭或未达到启动轮时只返回本批新题。
        return proposal, decision, items  # 不扫描历史题池，也不产生额外请求。
    pool = {item["origin_task_id"]: item for record in snapshot for item in record["tasks"] if item["group"] == "target"}  # 原始目标题去重，不收集重测。
    rng = random.Random(f"retest:{seed}:{planned}")  # 重测使用独立且稳定的随机流。
    for origin in rng.sample(sorted(pool), min(history_count, len(pool))):  # 候选不足时取尽。
        old = pool[origin]  # 保留原编号、颜色和预算，只更新本次运行标识。
        task = Task(f"r{planned}-q{len(items)}", old["family"], old["size"], old["edges"], old["colors"], old["budget"])  # 历史状态不进入 Solver 观察。
        items.append((task, "historical_retest", origin))  # 将奖励与当前提案隔离。
    return proposal, decision, items  # 返回已经生成的整批题。


def solve_round(client, output, index, batch, records, rollouts_per_task, completion_bonus, waited, check_producer):  # 两种调度共用同一奖励和记录逻辑。
    models = client.models  # 保存实际执行的策略标识。
    append_json(output / "timing.jsonl", {"round": index, "batch_wait_seconds": waited})  # 实测等待独立于确定性回放。
    proposal, decision, items = batch  # 已准备就绪的任务。
    identity = decision["decision_id"] if decision else None  # 来源只从实际动作读取。
    tasks = []  # 题目证书和角色信息。
    for task, group, origin in items:  # 为监控和审计绑定来源。
        owner = None if group == "historical_retest" else identity  # 避免重测替当前动作获奖。
        tasks.append(dict(asdict(task), group=group, origin_task_id=origin, decision_id=owner, certificate=audit(task)))  # 每道题有独立证书。
    schedule = {"round": index, "planned_round": index, "observation_round": decision["observation_round"] if decision else None, "decision_id": identity, "proposal": proposal, "tasks": tasks, "solver_model": models["solver"], "challenger_model": models["challenger"]}  # 按归属而非时间偏移关联。
    write_json(output / f"schedule-{index}.json", schedule)  # 求解前固定批次。
    episodes = []  # 本轮已观察的结果。
    for (task, group, origin), meta in zip(items, tasks):  # 组别不从题型推断。
        for attempt in range(rollouts_per_task):  # 每次从原始着色开始。
            def save(episode):  # 中断时保留已执行步骤及组别。
                episode.update(group=group, origin_task_id=origin, decision_id=meta["decision_id"], solver_model=models["solver"])  # 求解后才附加归属。
                episodes.append(episode)  # 完成或中断的实际尝试。
                append_json(output / "episodes.jsonl", episode)  # 立即留下轨迹。
            try:  # 失败轮也要保留计划量与实际量。
                episode = solve(task, client, f"{task.task_id}-e{attempt}", save, completion_bonus, keep_action=True, before_step=check_producer)  # Solver 只得到正常题面。
            except Exception:  # 中断的批次不结算挑战奖励。
                partial = measure(proposal, episodes, decision, index, tasks, rollouts_per_task, models["solver"], records)  # 分组报告已完成工作。
                write_json(output / f"partial-{index}.json", {"round": index, "decision_id": identity, "monitor": partial["monitor"]})  # 未完成评价不进入反馈。
                raise  # 停止新的任务。
            save(episode)  # 保存正常或动作无效的回合。
    metrics = measure(proposal, episodes, decision, index, tasks, rollouts_per_task, models["solver"], records)  # 只从实际提交的动作结算。
    record = dict(schedule, evidence_mode=client.evidence_mode, episodes=episodes, **metrics)  # 固定完整结果。
    append_json(output / "rounds.jsonl", record)  # 异常退出也保留已结束轮次。
    return record  # 调度器收到完整结果后才发布给出题方。


def run_curriculum(client, output, seed=42, rounds=2, rollouts_per_task=2, initial_size=2, completion_bonus=1.0, schedule_mode="sync", historical_tasks_per_round=0, historical_start_round=1):  # 运行双角色题目调度。
    if schedule_mode not in ("sync", "async", "prefetch_one"):  # 保留旧配置的单批预取名称。
        raise ValueError("schedule_mode must be sync or async")  # 出题前拒绝未知模式。
    if any(type(v) is not int or v < 0 for v in (historical_tasks_per_round, historical_start_round)):  # 题量与轮号都是非负整数。
        raise ValueError("historical count and start round must be nonnegative integers")  # 不接受浮点题数。
    if rounds < 2 or rollouts_per_task < 1 or not math.isfinite(completion_bonus) or completion_bonus < 0:  # 沿用有限预算要求。
        raise ValueError("invalid rounds, rollouts or completion bonus")  # 阻止不可评价的运行。
    output.mkdir(parents=True, exist_ok=True)  # 正式入口为每次实验建立独立目录。
    def build(index, snapshot):  # 两种模式的差异只在收到的快照与调用时机。
        count = historical_tasks_per_round if index >= historical_start_round else 0  # 从指定轮起每批额外抽取原题。
        return prepare(client, output, snapshot, index, seed, initial_size, count, client.models)  # 调用同一组生成器和重测规则。
    def consume(index, batch, records, waited, check_producer):  # 统一计算奖励并保存证据。
        return solve_round(client, output, index, batch, records, rollouts_per_task, completion_bonus, waited, check_producer)  # 整批独立求解再结算。
    schedule = run_sync if schedule_mode == "sync" else run_async  # 旧预取名称对应同样的单批滞后语义。
    return schedule(rounds, build, consume)  # 只有异步分支启动生产线程。
