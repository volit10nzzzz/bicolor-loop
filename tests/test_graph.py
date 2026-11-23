import unittest  # 使用标准库测试运行器，避免引入训练依赖。
from bicolor.graph import Task, audit, generate_task, flip, satisfied  # 导入待实现的真实环境。


class GraphTests(unittest.TestCase):  # 覆盖可以手算的环境行为。
    def setUp(self):  # 每个用例都从讨论中的六节点图开始。
        self.task = Task("manual", "bridge", 2, ((0, 1), (0, 2), (0, 3), (3, 4), (3, 5)), (0, 1, 1, 0, 1, 1), 3)  # A、D 同红，其余蓝。

    def test_exact_increment_and_solution(self):  # 防止奖励位置、负增益或累计分计算错误。
        colors = self.task.colors  # 保存初始颜色。
        scores = [satisfied(self.task.edges, colors)]  # 初始应有四条合格边。
        for node in (3, 4, 5):  # 依次翻转 D、E、F。
            colors = flip(colors, node)  # 执行真实动作。
            scores.append(satisfied(self.task.edges, colors))  # 记录整数计分避免浮点判断。
        self.assertEqual(scores, [4, 3, 4, 5])  # 核对完整手算轨迹。
        report = audit(self.task)  # 独立核验难度与可解性。
        self.assertEqual(report["min_flips"], 3)  # 防止题目虚报最短修复步数。
        self.assertTrue(report["all_first_moves_drop"])  # 初始所有动作都应掉分。

    def test_generated_pairs_are_matched_and_reproducible(self):  # 检查配对题构造及种子稳定性。
        for size in (2, 3, 4):  # 覆盖全部支持规模。
            bridge = generate_task("bridge", size, 17, "a")  # 生成必须退步的图。
            chain = generate_task("chain", size, 17, "b")  # 生成可以持平前进的图。
            first, second = audit(bridge), audit(chain)  # 让验证器独立核对两道题。
            self.assertEqual(first["min_flips"], size + 1)  # 两类题具有相同修复距离。
            self.assertEqual(first["min_flips"], second["min_flips"])  # 防止对照题难度标注错位。
            self.assertEqual(first["initial_satisfied"], second["initial_satisfied"])  # 配对初始分保持一致。
            self.assertTrue(first["all_first_moves_drop"])  # bridge 的退步属性成立。
            self.assertFalse(second["all_first_moves_drop"])  # chain 具有非下降首步。
            self.assertEqual(bridge, generate_task("bridge", size, 17, "a"))  # 同种子重建同题。

    def test_invalid_task_is_rejected(self):  # 模型或文件中的非法图不能变成正常测评题。
        invalid = Task("bad", "bridge", 2, ((0, 1), (1, 2), (2, 0)), (0, 0, 0), 3)  # 三角形无法二染色。
        with self.assertRaises(ValueError):  # 无解题必须明确报错。
            audit(invalid)  # 调用真实合法性检查。


if __name__ == "__main__":  # 支持单独执行测试文件。
    unittest.main()  # 输出测试结果并使用正确退出码。
