# -*- coding: UTF-8 -*-
"""================================================================
>> Project -> File  : act-main -> utils
>> IDE              : PyCharm
>> Author           : Wasve
>> Date             : 2024/12/18 18:31
>> Desc             : None
================================================================"""


import math
import time
import sys
from math import cos, sin, atan2, sqrt, acos, pi
import numpy as np

class BrokenExcavator:

    def __init__(self):
        self.d1 = 126
        self.a1 = -70.5
        self.alpha1 = math.pi / 2

        self.a2 = 746.7261
        self.alpha2 = 0
        self.offset2 = 0.5997814963354907

        self.a3 = 275
        self.alpha3 = 0
        self.offset3 = -1.3489559562060935

        self.a4 = 160.39
        self.alpha4 = 0
        self.offset4 = -0.8217

        self.px = None
        self.py = None
        self.pz = None

    def ik(self, _px, _py, _pz):
        self.px = _px
        self.py = _py
        self.pz = _pz
        return self.__ik()

    def __ik(self):
        theta1 = atan2(self.py, self.px)
        self.py = self.py - sin(theta1) * 35
        self.px = self.px - cos(theta1) * 35

        r = sqrt(self.px ** 2 + self.py ** 2) - self.a1  # 从关节1到末端的水平距离

        z = self.pz - self.d1 + self.a4  # 关节1到末端的垂直距离

        _R = sqrt(r ** 2 + z ** 2)
        theta2 = acos((self.a2 ** 2 + _R ** 2 - self.a3 ** 2) / (2 * self.a2 * _R)) + atan2(z, r)
        theta3 = acos((self.a2 ** 2 + self.a3 ** 2 - _R ** 2) / (2 * self.a2 * self.a3)) - pi
        theta4 = -pi / 2 - theta2 - theta3
        theta2 = theta2 - self.offset2
        theta3 = theta3 - self.offset3
        theta4 = theta4 - self.offset4
        return round(theta1, 4), round(theta2, 4), round(theta3, 4), round(theta4, 4)

    def fk(self, theta1, theta2, theta3, theta4):
        # 考虑偏移量
        theta2 += self.offset2
        theta3 += self.offset3
        theta4 += self.offset4

        # 定义 DH 参数转换矩阵
        def dh_transform(a, alpha, d, theta):
            return np.array([
                [math.cos(theta), -math.sin(theta) * math.cos(alpha), math.sin(theta) * math.sin(alpha),
                 a * math.cos(theta)],
                [math.sin(theta), math.cos(theta) * math.cos(alpha), -math.cos(theta) * math.sin(alpha),
                 a * math.sin(theta)],
                [0, math.sin(alpha), math.cos(alpha), d],
                [0, 0, 0, 1]
            ])

        # 构造各关节的变换矩阵
        T1 = dh_transform(self.a1, self.alpha1, self.d1, theta1)
        T2 = dh_transform(self.a2, self.alpha2, 0, theta2)
        T3 = dh_transform(self.a3, self.alpha3, 0, theta3)
        T4 = dh_transform(self.a4, self.alpha4, 0, theta4)

        # T1 = dh_transform(self.a1, self.alpha1, self.d1, theta1)
        # T2 = dh_transform(self.a2, 0, 0, theta2)
        # T3 = dh_transform(self.a3, 0, 0, theta3)
        # T4 = dh_transform(self.a4, 0, 0, theta4)

        # 计算总的变换矩阵
        T = T1 @ T2 @ T3 @ T4

        # 提取 px, py, pz 并应用 35 的偏移量
        px = T[0, 3] + 35 * math.cos(theta1)
        py = T[1, 3] + 35 * math.sin(theta1)
        pz = T[2, 3]
        return round(px, 4), round(py, 4), round(pz, 4)

ex = BrokenExcavator()
data = []
with open(r"E:\doctor\teaching_graps\act-main\target_pos_4.txt", 'r') as f:
    datas = f.readlines()
    for items in datas:
        items = items.split()

        data.append(ex.fk(float(items[0]), float(items[1]), float(items[2]), float(items[3])))

with open("target4_pos.txt", 'w') as f:
    for item in data:
        f.write(f"{item[0]} {item[1]} {item[2]}\n")
#