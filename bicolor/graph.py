from dataclasses import dataclass  # 用不可变数据类保存题目。
from itertools import product  # 穷举小图着色，独立核验最短修复距离。
import random  # 用局部随机数生成器复现题目。


@dataclass(frozen=True)  # 运行中不允许原地改变初始题目。
class Task:  # 一道可验证的二染色修复题。
    task_id: str  # 轨迹与 Challenger 决策使用的题目标识。
    family: str  # 题型供任务构造、课程与分析使用。
    size: int  # 构造规模，取值为 2 到 4。
    edges: tuple[tuple[int, int], ...]  # 无向边集合，节点使用连续整数编号。
    colors: tuple[int, ...]  # 初始颜色，0 和 1 分别表示两种颜色。
    budget: int  # 允许翻转的最大次数。

    def public(self):  # 提供 Solver 能观察到的题目字段。
        return {"task_id": self.task_id, "edges": self.edges, "initial_colors": self.colors, "budget": self.budget}  # 隐藏难度证书与最优距离。


def satisfied(edges, colors):  # 直接统计两端异色的边数。
    return sum(colors[left] != colors[right] for left, right in edges)  # 整数计数是所有评分的依据。


def flip(colors, node):  # 执行一次节点翻色，返回新状态。
    if type(node) is not int or not 0 <= node < len(colors):  # API 动作必须引用存在的整数节点。
        raise ValueError("node must be an existing integer node ID")  # 非法动作交由调用者记录。
    updated = list(colors)  # 保留输入状态以便计算前后差。
    updated[node] = 1 - updated[node]  # 二值颜色翻转。
    return tuple(updated)  # 使用不可变状态避免历史被覆盖。


def audit(task):  # 对小图穷举，产生仅供验证与分析使用的证书。
    count = len(task.colors)  # 获取节点数量。
    if not 2 <= count <= 10 or not task.edges:  # 穷举范围由实验支持规模限定。
        raise ValueError("supported tasks need 2..10 nodes and at least one edge")  # 拒绝无法按约定验证的题目。
    if any(type(color) is not int or color not in (0, 1) for color in task.colors):  # 确保颜色属于二值状态空间。
        raise ValueError("colors must contain only integer 0 or 1")  # 避免错误颜色被当作第三种颜色。
    normalized = {tuple(sorted(edge)) for edge in task.edges}  # 规范化无向边以检查重复。
    if len(normalized) != len(task.edges) or any(a == b or a < 0 or b >= count for a, b in normalized):  # 重复边或非法节点会扭曲得分。
        raise ValueError("edges must be unique, valid, and without self loops")  # 明确拒绝无效输入。
    goals = [state for state in product((0, 1), repeat=count) if satisfied(task.edges, state) == len(task.edges)]  # 独立枚举所有满分着色。
    if not goals:  # 无解题无法承担当前的满分目标。
        raise ValueError("graph is not two-colorable")  # 无解任务不进入 Solver。
    distance = min(sum(a != b for a, b in zip(task.colors, goal)) for goal in goals)  # 自由翻转下最短距离等于最小汉明距离。
    if distance == 0 or task.budget < distance:  # 过滤已解题与预算内不可解题。
        raise ValueError("task must be unsolved and solvable within the budget")  # 保证挑战有有效求解空间。
    initial = satisfied(task.edges, task.colors)  # 保存当前整数分数。
    all_drop = all(satisfied(task.edges, flip(task.colors, node)) < initial for node in range(count))  # 验证是否每种首步都会掉分。
    return {"min_flips": distance, "initial_satisfied": initial, "edge_count": len(task.edges), "all_first_moves_drop": all_drop}  # 返回不含具体解路径的证书。


def generate_task(family, size, seed, task_id):  # 按种子构造一道人为控制难度的题目。
    if family not in ("bridge", "chain") or type(size) is not int or not 2 <= size <= 4:  # 校验 Challenger 能控制的有限动作空间。
        raise ValueError("family must be bridge/chain and size an integer in 2..4")  # 错误提案不静默替换。
    count = 2 * size + 2  # 两类题使用相同节点数。
    middle = size + 1  # 图中两个半区的分界位置。
    if family == "bridge":  # 两个带叶子的中心由一条边连接。
        edges = [(0, middle)] + [(0, node) for node in range(1, middle)] + [(middle, node) for node in range(middle + 1, count)]  # 构造相同边数的树。
        colors = [0 if node in (0, middle) else 1 for node in range(count)]  # 只让中心连接边冲突。
    else:  # 链形题提供可以持平移动冲突的对照。
        edges = [(node, node + 1) for node in range(count - 1)]  # 相邻节点构成一条链。
        colors = [(node % 2) ^ int(node >= middle) for node in range(count)]  # 翻转后半段，制造唯一的中间冲突。
    rng = random.Random(seed)  # 生成器不影响全局随机状态。
    labels = list(range(count))  # 准备随机重命名节点。
    rng.shuffle(labels)  # 避免固定字母位置成为提示。
    inversion = rng.randrange(2)  # 随机交换两种颜色的语义。
    renamed = [0] * count  # 为新编号分配颜色。
    for old, new in enumerate(labels):  # 同时变换边和颜色以保持题目同构。
        renamed[new] = colors[old] ^ inversion  # 按新编号写入颜色。
    renamed_edges = tuple(sorted(tuple(sorted((labels[a], labels[b]))) for a, b in edges))  # 规范化边顺序。
    task = Task(task_id, family, size, renamed_edges, tuple(renamed), size + 1)  # 步数预算等于该构造的最短修复距离。
    certificate = audit(task)  # 使用独立枚举器核验构造结果。
    if certificate["min_flips"] != size + 1 or certificate["all_first_moves_drop"] != (family == "bridge"):  # 验证生成器承诺的难度条件。
        raise ValueError("generated task does not satisfy its difficulty contract")  # 构造错误不能流入实验。
    return task  # 返回通过验证的任务。
