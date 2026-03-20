'''
输出丑陋的向量处理
'''
import json
import math
import numpy as np
class VectorTaskProcessor:
    def __init__(self, index):
        """初始化向量任务处理器"""
        with open('data.json', 'r') as file:
            data = json.load(file)
        data = data[index]
        # print(data)
        # 打印组名标题
        print(f"### current process:{data['group_name']} ###")
        self.vector = data['vectors']
        self.ori_axis = data['ori_axis']
        self.tasks = data['tasks']
        # print(self.vector)
        # print(self.tasks)
        # print(self.ori_axis)
    def _print(self):
        print(self.vector)
    def __axis_cos_angle(self, vec):
        """处理轴夹角计算任务"""
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
    def area(self):
        """计算基向量构成的面积/体积"""
        return np.linalg.det(self.ori_axis)
    def _axis_angle(self, vec):
        """计算向量与各坐标轴的夹角（弧度）"""
        angles = self.__axis_cos_angle(vec)
        res = []
        for angle in angles:
            res.append(math.acos(angle))
        return angles
    def _axis_projection(self, vec):
        """计算向量在各坐标轴上的投影"""
        res = []
        norm_v = np.linalg.norm(vec)
        for angle in self.__axis_cos_angle(vec):
            res.append(norm_v * angle)
        return res
    def _change_axis(self, target):
        """改变坐标系"""
        # check linear relationship
        if np.linalg.det(target) < 1e-10:
            print("linear Error\n")
            return None
        try:
            vectors = self.vector
            new_vectors = []
            for vector in vectors:
                # 归一化处理
                norms = np.linalg.norm(self.ori_axis, axis=1)
                vector /= norms
                # 坐标系转换
                new = np.linalg.inv(target) @ (self.ori_axis @ vector)
                new_vectors.append(new)
            new_vectors = np.array(new_vectors)
            norms = np.linalg.norm(target, axis=1)
            self.vector = new_vectors * norms
            # print(self.vector)
            self.ori_axis = target
        except np.linalg.LinAlgError as e:
            print(f"{e}\n")
        return 1
    def process_task(self):
        """处理所有任务"""
        for task in self.tasks:
            match task['type']:
                case 'axis_angle':
                    # continue
                    print(f"--- calculate angle for every axis ---")
                    for vector in self.vector:
                        angle = self._axis_angle(vector)
                        print(f"vector:{vector} angle:{angle}")
                case 'change_axis':
                    # continue
                    print(f"--- change axis ---")
                    self._change_axis(task['obj_axis'])
                    for vector in self.vector:
                        print(f"changed vector:{vector}")
                case 'area':
                    print(f"--- calculate area ---")
                    print(f"area:{self.area()}")
                case 'axis_projection':
                    # continue
                    print(f"--- calculate vectors' projection for every axis ---")
                    for vector in self.vector:
                        res = self._axis_projection(vector)
                        print(f"vector:{vector} projection:{res}")

                case _:
                    print(f"None matched task!")
                    return 0
        return 1
test = VectorTaskProcessor(1)
test.process_task()
