'''
输出漂亮的向量处理
'''

import json
import math
import numpy as np
from typing import List, Union, Optional


class Color:
    """ANSI颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    # 背景色
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


class VectorTaskProcessor:
    def __init__(self, index: int):
        """初始化向量任务处理器"""
        with open('data.json', 'r') as file:
            data = json.load(file)
        data = data[index]

        # 打印组名标题
        self._print_header(f"当前处理: {data['group_name']}")

        self.vector = data['vectors']
        self.ori_axis = data['ori_axis']
        self.tasks = data['tasks']

        # 打印基本信息
        self._print_basic_info()

    def _print_header(self, title: str, width: int = 60):
        """打印标题"""
        print(f"\n{Color.BG_BLUE}{Color.BOLD}{'=' * width}{Color.END}")
        print(f"{Color.BG_BLUE}{Color.BOLD}{title.center(width)}{Color.END}")
        print(f"{Color.BG_BLUE}{Color.BOLD}{'=' * width}{Color.END}\n")

    def _print_section(self, title: str, width: int = 50):
        """打印章节标题"""
        print(f"\n{Color.CYAN}{Color.BOLD}╔{'═' * (width - 2)}╗{Color.END}")
        print(f"{Color.CYAN}{Color.BOLD}║ {title.center(width - 4)} ║{Color.END}")
        print(f"{Color.CYAN}{Color.BOLD}╚{'═' * (width - 2)}╝{Color.END}")

    def _print_basic_info(self):
        """打印基本信息"""
        print(f"{Color.YELLOW}【基本信息】{Color.END}")
        print(f"{Color.GREEN}• 原始基向量:{Color.END}")
        for i, axis in enumerate(self.ori_axis):
            print(f"  轴{i + 1}: {self._format_vector(axis)}")

        print(f"\n{Color.GREEN}• 待处理向量 ({len(self.vector)}个):{Color.END}")
        for i, vec in enumerate(self.vector):
            print(f"  向量{i + 1}: {self._format_vector(vec)}")

        print(f"\n{Color.GREEN}• 任务列表 ({len(self.tasks)}个):{Color.END}")
        for i, task in enumerate(self.tasks):
            print(f"  {i + 1}. {task['type']}")

    def _format_vector(self, vec: Union[np.ndarray, List]) -> str:
        """格式化向量输出"""
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
        if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], (int, float, np.floating)):
            # 标量或一维向量
            formatted = [f"{x:8.4f}" for x in vec]
            return f"[{', '.join(formatted)}]"
        elif isinstance(vec, list) and len(vec) > 0:
            # 多维数组
            rows = []
            for row in vec:
                if isinstance(row, (list, np.ndarray)):
                    formatted = [f"{x:8.4f}" for x in row]
                    rows.append(f"[{', '.join(formatted)}]")
            return '\n    '.join(rows)
        return str(vec)

    def _format_angle(self, rad: float) -> str:
        """格式化角度输出（弧度转度数）"""
        deg = math.degrees(rad)
        return f"{rad:8.4f} rad ({deg:8.2f}°)"

    def _axis_cos_angle(self, vec: np.ndarray) -> List[float]:
        """计算向量与各坐标轴夹角的余弦值"""
        angles = []
        for axis in self.ori_axis:
            axis_dot = np.dot(vec, axis)
            norm_sum = np.linalg.norm(axis) * np.linalg.norm(vec)
            if norm_sum == 0:
                res = 0
            else:
                res = axis_dot / norm_sum
            angles.append(res)
        return angles

    def area(self) -> float:
        """计算基向量构成的面积/体积"""
        return np.linalg.det(self.ori_axis)

    def _axis_angle(self, vec: np.ndarray) -> List[float]:
        """计算向量与各坐标轴的夹角（弧度）"""
        cos_angles = self._axis_cos_angle(vec)
        angles = [math.acos(angle) for angle in cos_angles]
        return angles

    def _axis_projection(self, vec: np.ndarray) -> List[float]:
        """计算向量在各坐标轴上的投影"""
        res = []
        norm_v = np.linalg.norm(vec)
        cos_angles = self._axis_cos_angle(vec)
        for cos_angle in cos_angles:
            res.append(norm_v * cos_angle)
        return res

    def _change_axis(self, target: np.ndarray) -> Optional[int]:
        """改变坐标系"""
        # 检查线性相关性
        if np.linalg.det(target) < 1e-10:
            print(f"{Color.RED}❌ 错误: 目标基向量线性相关，无法构成新坐标系{Color.END}")
            return None

        try:
            vectors = self.vector.copy()
            new_vectors = []

            print(f"{Color.YELLOW}坐标变换详情:{Color.END}")
            print(f"  {Color.GREEN}原基向量 → 新基向量{Color.END}")
            for i, axis in enumerate(self.ori_axis):
                print(f"  轴{i + 1}: {self._format_vector(axis)} → {self._format_vector(target[i])}")

            for i, vector in enumerate(vectors):
                # 归一化处理
                norms = np.linalg.norm(self.ori_axis, axis=1)
                vector_normalized = vector / norms

                # 坐标变换
                new_vec = np.linalg.inv(target) @ (self.ori_axis @ vector_normalized)
                new_vectors.append(new_vec)

                print(f"\n  {Color.GREEN}向量{i + 1}变换:{Color.END}")
                print(f"    原坐标: {self._format_vector(vector)}")
                print(f"    归一化: {self._format_vector(vector_normalized)}")
                print(f"    新坐标: {self._format_vector(new_vec)}")

            new_vectors = np.array(new_vectors)
            norms = np.linalg.norm(target, axis=1)
            self.vector = new_vectors * norms
            self.ori_axis = target

            print(f"\n{Color.GREEN}✓ 坐标变换完成{Color.END}")
            return 1

        except np.linalg.LinAlgError as e:
            print(f"{Color.RED}❌ 线性代数错误: {e}{Color.END}")
            return None

    def process_task(self) -> int:
        """处理所有任务"""
        print(f"\n{Color.BG_GREEN}{Color.BOLD}{' 开始处理任务 ':=^60}{Color.END}\n")

        for i, task in enumerate(self.tasks):
            task_type = task['type']
            self._print_section(f"任务 {i + 1}: {task_type}", 50)

            match task_type:
                case 'axis_angle':
                    self._process_axis_angle()

                case 'change_axis':
                    self._process_change_axis(task['obj_axis'])

                case 'area':
                    self._process_area()

                case 'axis_projection':
                    self._process_axis_projection()

                case _:
                    print(f"{Color.RED}⚠️  未知任务类型: {task_type}{Color.END}")
                    return 0

        print(f"\n{Color.BG_GREEN}{Color.BOLD}{' 所有任务完成 ':=^60}{Color.END}\n")
        return 1

    def _process_axis_angle(self):
        """处理轴夹角计算任务"""
        print(f"{Color.BLUE}计算各向量与坐标轴的夹角:{Color.END}")
        print(f"{'向量':<20} {'与轴1夹角':<25} {'与轴2夹角':<25} {'与轴3夹角':<25}")
        print(f"{'-' * 100}")

        for i, vector in enumerate(self.vector):
            angles = self._axis_angle(vector)
            angle_strs = [self._format_angle(angle) for angle in angles]

            print(f"{Color.YELLOW}向量{i + 1}: {self._format_vector(vector)}{Color.END}")
            for j, angle_str in enumerate(angle_strs):
                print(f"  {Color.GREEN}轴{j + 1}{Color.END}: {angle_str}")
            print()

    def _process_change_axis(self, target_axis):
        """处理坐标变换任务"""
        print(f"{Color.BLUE}执行坐标变换:{Color.END}")
        print(f"{Color.YELLOW}目标基向量:{Color.END}")
        for j, axis in enumerate(target_axis):
            print(f"  轴{j + 1}: {self._format_vector(axis)}")
        print()

        result = self._change_axis(target_axis)
        if result:
            print(f"\n{Color.GREEN}变换后向量:{Color.END}")
            for j, vector in enumerate(self.vector):
                print(f"  向量{j + 1}: {self._format_vector(vector)}")

    def _process_area(self):
        """处理面积计算任务"""
        print(f"{Color.BLUE}计算基向量构成的面积/体积:{Color.END}")
        area_value = self.area()
        print(f"\n{Color.YELLOW}基向量行列式:{Color.END}")
        for j, axis in enumerate(self.ori_axis):
            print(f"  轴{j + 1}: {self._format_vector(axis)}")

        print(f"\n{Color.GREEN}面积/体积 = {area_value:.6f}{Color.END}")
        print(f"{Color.GREEN}绝对面积 = {abs(area_value):.6f}{Color.END}")

    def _process_axis_projection(self):
        """处理轴投影计算任务"""
        print(f"{Color.BLUE}计算各向量在坐标轴上的投影:{Color.END}")
        print(f"{'向量':<20} {'在轴1投影':<20} {'在轴2投影':<20} {'在轴3投影':<20}")
        print(f"{'-' * 85}")

        for i, vector in enumerate(self.vector):
            projections = self._axis_projection(vector)
            proj_strs = [f"{proj:8.4f}" for proj in projections]

            print(f"{Color.YELLOW}向量{i + 1}: {self._format_vector(vector)}{Color.END}")
            for j, proj_str in enumerate(proj_strs):
                print(f"  {Color.GREEN}轴{j + 1}{Color.END}: {proj_str}")
            print()


# 测试代码
if __name__ == "__main__":
    print(f"{Color.BG_CYAN}{Color.BOLD}{' 向量任务处理器启动 ':=^60}{Color.END}")

    try:
        test = VectorTaskProcessor(10)
        test.process_task()

    except FileNotFoundError:
        print(f"{Color.RED}❌ 错误: 找不到 data.json 文件{Color.END}")
    except (KeyError, IndexError) as e:
        print(f"{Color.RED}❌ 数据格式错误: {e}{Color.END}")
    except Exception as e:
        print(f"{Color.RED}❌ 未知错误: {e}{Color.END}")

    print(f"\n{Color.BG_CYAN}{Color.BOLD}{' 程序执行结束 ':=^60}{Color.END}")
