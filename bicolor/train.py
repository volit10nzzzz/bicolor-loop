import argparse  # 提供训练数据准备入口。
import json  # 读写可审计轨迹。
from .replay import verify_run  # 导出前校验状态与双方奖励。
from .loop import solver_observation  # 与实际发给模型的观察共用构造函数。
from pathlib import Path  # 管理运行与导出路径。


def export_transitions(run_dir, output, role="solver"):  # 仅准备训练输入，不启动任何优化器。
    verify_run(run_dir)  # 缺失或被篡改的证据不能进入训练接口。
    if role not in ("solver", "challenger"):  # 外部角色参数仅接受两种契约。
        raise ValueError("unsupported training role")  # 不静默导出其他数据。
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))  # 沿用数据来源标记。
    count, skipped = 0, 0  # 记录有效转移与跳过的无效动作回合。
    if role == "challenger":  # 出题方每个提案对应一个延迟奖励样本。
        records = [json.loads(line) for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]  # 按轮读取已验证记录。
        with output.open("x", encoding="utf-8") as stream:  # 不覆盖已有训练数据。
            for record in records:  # 奖励按实际提案关联，可支持预取。
                evaluation = record["challenger_reward"]  # 从执行该批题的轮次取延迟评价。
                if evaluation is None:  # 种子批没有需要训练的出题动作。
                    continue  # 不计入缺失评价的提案数。
                if evaluation["status"] != "scored":  # 缺少有效评价时不伪造训练目标。
                    skipped += 1  # 报告排除的提案数。
                    continue  # 保留未判定含义。
                decision = json.loads((run_dir / f"{evaluation['decision_id']}.json").read_text(encoding="utf-8"))  # 读取原始观察和提案。
                row = {"role": role, "evidence_mode": manifest["evidence_mode"], "decision_id": evaluation["decision_id"], "observation": decision["input"], "action": decision["output"], "reward": evaluation["reward"], "evaluation": evaluation, "terminated": True, "truncated": False}  # 独立单次出题任务的延迟回报契约。
                if manifest["schema_version"] >= 3:  # 历史版本只记录挑战奖励的标量。
                    row.update(challenger_model=decision["challenger_model"], challenge_reward=evaluation["challenge_reward"], diagnostic_quality=evaluation["diagnostic_quality"])  # 两个监督分量保留各自数值或未知值。
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")  # 不伪造 token 概率或优化器状态。
                count += 1  # 统计真实导出数量。
        return {"status": "export_only", "role": role, "transitions": count, "skipped_decisions": skipped, "optimizer_started": False}  # 明确仅数据准备。
    with output.open("x", encoding="utf-8") as stream:  # 禁止覆盖已有导出。
        for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines():  # 逐条读取完整回合。
            episode = json.loads(line)  # 恢复回合数据。
            if episode["status"] in ("invalid_action", "interrupted"):  # 未定义无效动作训练编码或中断 bootstrap。
                skipped += 1  # 明确报告排除范围。
                continue  # 不悄悄改变失败回合的终止语义。
            for index, step in enumerate(episode["steps"]):  # 每个有效动作对应一个转移。
                terminal = index == len(episode["steps"]) - 1  # 有限步数任务把预算终止视为终局。
                row = {"role": role, "delta_reward": step["delta_reward"], "completion_bonus": step["completion_bonus"], "episode_id": episode["episode_id"], "evidence_mode": manifest["evidence_mode"], "task": episode["task"], "state": {"colors": step["before"], "remaining_steps": step["remaining_steps_before"]}, "history": episode["steps"][:index], "action": {"node": step["node"]}, "reward": step["reward"], "next_state": {"colors": step["after"], "remaining_steps": step["remaining_steps_before"] - 1}, "terminated": terminal, "truncated": False, "terminal_status": episode["status"] if terminal else None}  # 保留预算和历史以还原模型观察。
                if manifest["schema_version"] >= 3:  # 新数据保留实际观察、原始回答和模型版本。
                    score_before = step["satisfied_before"] / len(episode["task"]["edges"])  # 仅从动作之前的环境状态恢复分数。
                    row["observation"] = solver_observation(episode["episode_id"], episode["task"], step["before"], score_before, step["remaining_steps_before"], row["history"], manifest["config"]["run"]["completion_bonus"])  # 历史在本动作之前截断，重测不夹带旧轨迹。
                    row.update(group=episode["group"], origin_task_id=episode["origin_task_id"], decision_id=episode["decision_id"], solver_model=episode["solver_model"], raw_action=step["raw_action"])  # 分组仅作训练样本元数据，完整回答与执行节点分开。
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")  # 输出环境级转移，未伪造 token 概率。
                count += 1  # 更新导出数量。
    return {"status": "export_only", "transitions": count, "skipped_episodes": skipped, "optimizer_started": False}  # 明确训练尚未实现。


def main():  # 预留未来训练接入的数据入口。
    parser = argparse.ArgumentParser(description="仅导出训练转移；不执行梯度更新")  # 避免误解为训练命令。
    parser.add_argument("--run", required=True, type=Path)  # 指定完整且可回放核验的运行目录。
    parser.add_argument("--output", required=True, type=Path)  # 指定新的 JSONL 文件。
    parser.add_argument("--role", choices=("solver", "challenger"), default="solver")  # 两个角色独立导出。
    args = parser.parse_args()  # 解析参数。
    print(json.dumps(export_transitions(args.run, args.output, args.role), ensure_ascii=False))  # 展示真实导出状态。


if __name__ == "__main__":  # 支持 python -m bicolor.train。
    main()  # 仅启动数据导出。
