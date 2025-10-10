# -*- coding: UTF-8 -*-
"""================================================================
>> Project -> File  : act-main -> service_func
>> IDE              : PyCharm
>> Author           : Wasve
>> Date             : 2024/10/11 16:28
>> Desc             : None
================================================================"""
import math
import time
import sys
from math import cos, sin, atan2, sqrt, acos, pi
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
# from matplotlib import pyplot3 as plt
from spatialmath import SE3
from wandb.old.summary import h5py
import cv2 as cv3
import sim
import random
from MainWindow import Ui_Form
from roboticstoolbox import DHRobot, RevoluteDH


# excavator = DHRobot([RevoluteDH(d=126, a=-70.5, alpha=np.pi / 2, offset=0),
#                      RevoluteDH(d=0, a=746.7261, alpha=0, offset=0.5997814963354907),
#                      RevoluteDH(d=0, a=275, alpha=0, offset=-1.3489559562060935),
#                      RevoluteDH(d=0, a=164.78, alpha=0, offset=-0.8217)], name='Excavator')
#
# excavator.tool = SE3([[1, 0, 0, 0],
#                       [0, 1, 0, 35.0],
#                       [0, 0, 1, 0],
#                       [0, 0, 0, 1]])
# a: SE3 = excavator.fkine([0, 0, 0, 0])
#


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


def add_rock_objects(client, _name, _num, _part_num, _path, _pos, _orient, _color, _scale):
    _ = sim.simxCallScriptFunction(client,
                                   "remoteApiCommandServer",
                                   sim.sim_scripttype_childscript,
                                   'addRock',
                                   [_num, _part_num],
                                   _pos + _orient + _color + [_scale],
                                   [_name, _path],
                                   bytearray(),
                                   sim.simx_opmode_blocking)


class InterpPath:

    def __init__(self):
        self.path = None
        self.delta = 0.005
        self.num = 30
        self._idx = -1
        self.interp = []
        self.lvp_list = []
        self.velocity_plan = False

    def __iter__(self):
        return self

    def set_p1_p2(self, p1, p2):
        self._idx = -1
        self.set_path([p1, p2])

    def set_p1_p2_p3(self, p1, p2, p3):
        self.set_path([p1, p2, p3])

    def set_path(self, _path):
        self.path = _path
        nums = len(self.path)
        interp_list = []
        delta_list = []
        self.interp = []
        self.interp.append(self.path[0])
        for _i in range(1, nums):
            _delta = np.array(self.path[_i]) - np.array(self.path[_i - 1])
            _mini_delta = _delta / self.num
            interp_list.append(self.num)
            delta_list.append(_mini_delta)
        for _step, _delta_ in zip(interp_list, delta_list):
            for __step in range(_step):
                self.interp.append(_delta_ + self.interp[-1])

    def __next__(self):
        if self.velocity_plan:
            self._idx += 1
            if self._idx < len(self.lvp_list):
                return self.lvp_list[self._idx]
            else:
                raise StopIteration
        else:
            self._idx += 1
            if self._idx < len(self.interp):
                return self.interp[self._idx]
            else:
                raise StopIteration

    def __len__(self):
        return len(self.interp)


class InterFace(QWidget):
    ui = Ui_Form()

    def __init__(self):
        super(QWidget, self).__init__()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon(".\\window_icon.png"))
        self.setWindowTitle("hdf5 generator")
        self.client = None
        self.camera_names = ['top', 'ttop', 'left', 'focus']

        self.work_pos = [0.0, 0.0, 0.0]
        self.target_pos = [0.0, 0.0, 0.0]
        self.interp = InterpPath()
        self.time = QTimer(self)
        self.time.start(2000)
        self.time.timeout.connect(self._update_pos)

        self.count = 0
        self.joint_traj = []
        self.is_savable = True
        self.data_dict = {'/observations/qpos': [],
                          '/action': []}

        for cam_name in self.camera_names:
            self.data_dict[f'/observations/images/{cam_name}'] = []
        self.total_step = 0
        self.ui.connect_sim.clicked.connect(self.connect_sim_func)
        self.ui.start_sim.clicked.connect(self.start_sim_func)
        self.ui.pause_sim.clicked.connect(self._reset)
        self.ui.stop_sim.clicked.connect(self.stop_sim_func)
        self.ui.save_h5py.clicked.connect(self._save_h5py)
        self.ui.dummy_x.valueChanged.connect(self.target_x_changed)
        self.ui.dummy_y.valueChanged.connect(self.target_y_changed)
        self.ui.dummy_z.valueChanged.connect(self.target_z_changed)
        self.ui.tip_x.valueChanged.connect(self.tip_x_changed)
        self.ui.tip_y.valueChanged.connect(self.tip_y_changed)
        self.ui.tip_z.valueChanged.connect(self.tip_z_changed)
        self.ui.move_to.clicked.connect(self.move_to)
        self.ui.move_tox.textChanged.connect(self.target_x_changed)
        self.ui.move_toy.textChanged.connect(self.target_y_changed)
        self.ui.move_toz.textChanged.connect(self.target_z_changed)
        # self.ui.tip_x_label.windowTitleChanged.connect(self.slide_x_update)
        # self.ui.tip_y_label.windowTitleChanged.connect(self.slide_y_update)
        # self.ui.tip_z_label.windowTitleChanged.connect(self.slide_z_update)
        self.ui.add_rock.clicked.connect(self.add_rocks)
        self.ui.count.textChanged.connect(self.update_count)
        self.ui.go_home.clicked.connect(self._go_home)
        self.delta = float(self.ui.delta.text())

    def _go_home(self):
        if self.client is not None:
            self._set_pos(-0.03525565, 0.250978, -0.44087, 0.190066)

    def get_current_image(self):
        _, resolution, _img_top = sim.simxGetVisionSensorImage(self.client, self.top_camera, 0,
                                                               sim.simx_opmode_blocking)
        _, resolution, _img_focus = sim.simxGetVisionSensorImage(self.client, self.focus_camera, 0,
                                                                 sim.simx_opmode_blocking)
        _, resolution, _img_left = sim.simxGetVisionSensorImage(self.client, self.left_camera, 0,
                                                                sim.simx_opmode_blocking)
        # _, ret, _img_right = sim.simxGetVisionSensorImage(self.client, self.right_camera, 0, sim.simx_opmode_blocking)
        _, resolution, _img_ttop = sim.simxGetVisionSensorImage(self.client, self.ttop_camera, 0,
                                                                sim.simx_opmode_blocking)

        sim_ret, resolution, depth_buffer1 = sim.simxGetVisionSensorDepthBuffer(self.client, self.top_camera,
                                                                                sim.simx_opmode_blocking)
        sim_ret, resolution, depth_buffer2 = sim.simxGetVisionSensorDepthBuffer(self.client, self.left_camera,
                                                                                sim.simx_opmode_blocking)
        sim_ret, resolution, depth_buffer3 = sim.simxGetVisionSensorDepthBuffer(self.client, self.left_camera,
                                                                                sim.simx_opmode_blocking)
        sim_ret, resolution, depth_buffer4 = sim.simxGetVisionSensorDepthBuffer(self.client, self.ttop_camera,
                                                                                sim.simx_opmode_blocking)
        depth_img1 = np.asarray(depth_buffer1)
        depth_img1.shape = (resolution[1], resolution[0])
        depth_img1 = np.fliplr(depth_img1)
        zNear = 0.01
        zFar = 5
        depth_img1 = depth_img1 * (zFar - zNear) + zNear

        depth_img2 = np.asarray(depth_buffer2)
        depth_img2.shape = (resolution[1], resolution[0])
        depth_img2 = np.fliplr(depth_img2)
        zNear = 0.01
        zFar = 5
        depth_img2 = depth_img2 * (zFar - zNear) + zNear

        depth_img3 = np.asarray(depth_buffer3)
        depth_img3.shape = (resolution[1], resolution[0])
        depth_img3 = np.fliplr(depth_img3)
        zNear = 0.01
        zFar = 5
        depth_img3 = depth_img3 * (zFar - zNear) + zNear

        depth_img4 = np.asarray(depth_buffer4)
        depth_img4.shape = (resolution[1], resolution[0])
        depth_img4 = np.fliplr(depth_img4)
        zNear = 0.01
        zFar = 5
        depth_img4 = depth_img4 * (zFar - zNear) + zNear

        _img_top = np.asarray(_img_top)
        _img_top.shape = (512, 512, 3)
        _img_top = _img_top.astype(np.uint8)
        _img_top[_img_top < 0] += 255
        _img_top = np.fliplr(_img_top)

        _img_focus = np.asarray(_img_focus)
        _img_focus.shape = (512, 512, 3)
        _img_focus = _img_focus.astype(np.uint8)
        _img_focus[_img_focus < 0] += 255
        _img_focus = np.fliplr(_img_focus)

        _img_left = np.asarray(_img_left)
        _img_left.shape = (512, 512, 3)
        _img_left = _img_left.astype(np.uint8)
        _img_left[_img_left < 0] += 255
        _img_left = np.fliplr(_img_left)

        _img_ttop = np.asarray(_img_ttop)
        _img_ttop.shape = (512, 512, 3)
        _img_ttop = _img_ttop.astype(np.uint8)
        _img_ttop[_img_ttop < 0] += 255
        _img_ttop = np.fliplr(_img_ttop)

        # if self.ui.is_save.isChecked():
            # cv.imshow("_img_top", _img_top)
            # cv.imshow("_img_ttop", _img_ttop)
            # cv.imshow("_img_left", _img_left)
            # cv.imshow("_img_focus", _img_focus)
            # cv.imshow("depth1", depth_img1)
            # cv.imshow("depth2", depth_img2)
            # cv.waitKey(0)

        return [_img_top, _img_ttop, _img_left, _img_focus], [depth_img1, depth_img2, depth_img3, depth_img4]

    def update_count(self, _v):
        self.count = int(_v)

    def _reset(self):
        self.data_dict = {'/observations/qpos': [],
                          '/action': []}
        for cam_name in self.camera_names:
            self.data_dict[f'/observations/images/{cam_name}'] = []
        self.stop_sim_func()
        time.sleep(2)
        self.start_sim_func()
        self._env_set_up()
        self.joint_traj = []
        self.count += 1
        self.total_step = 0
        self.is_savable = True
        self.ui.status.setText("reset simulation successful!!")
        self.ui.Tstep.setText(f"{self.total_step}")

    def add_rocks(self):

        for ii in range(4):
            add_rock_objects(self.client,
                             "block",
                             ii,
                             4,
                             fr"rock\auto_rock_{random.randint(1, 10)}_",
                             _pos=[0.1, np.random.uniform(-0.2, 0.2), 0.4],
                             _orient=[0.0, 0.0, 0.0],
                             _color=[152 / 255, 160 / 255, 169 / 255],
                             _scale=random.uniform(0.04, 0.045))
            time.sleep(2)

    def move_to(self):
        _, p2 = sim.simxGetObjectPosition(self.client,
                                          self.target_dummy,
                                          self.base_dummy,
                                          sim.simx_opmode_blocking)
        _, p1 = sim.simxGetObjectPosition(self.client,
                                          self.tip_handle,
                                          self.base_dummy,
                                          sim.simx_opmode_blocking)
        print("p1", p1)
        print("p2", p2)
        self.interp.num = int(self.ui.move_to_count.text())
        self.interp.set_p1_p2(p1, p2)
        float_packs = [point for points in self.interp.interp for point in points]
        packed_data = sim.simxPackFloats(float_packs)
        sim.simxSetStringSignal(self.client, 'coordinates_signal', packed_data, sim.simx_opmode_blocking)
        for idx, _pos in enumerate(self.interp):
            _joint = self.set_work_pos(_pos)
            if self.ui.is_save.isChecked() and idx != 0:
                self.joint_traj.append(_joint)
                self.total_step += 1
                self.ui.Tstep.setText(f"{self.total_step}")
                self.data_dict['/observations/qpos'].append(np.array(self._get_current_pos()))
                self.data_dict['/action'].append(np.array(_joint))
                iss, dss = self.get_current_image()
                i1, i2, i3, i4 = iss
                d1, d2, d3, d4 = dss
                self.data_dict[f'/observations/images/{self.camera_names[0]}'].append(i1)
                self.data_dict[f'/observations/images/{self.camera_names[1]}'].append(i2)
                self.data_dict[f'/observations/images/{self.camera_names[2]}'].append(i3)
                self.data_dict[f'/observations/images/{self.camera_names[3]}'].append(i4)
                self.data_dict[f'/observations/depths/{self.camera_names[0]}'].append(d1)
                self.data_dict[f'/observations/depths/{self.camera_names[1]}'].append(d2)
                self.data_dict[f'/observations/depths/{self.camera_names[2]}'].append(d3)
                self.data_dict[f'/observations/depths/{self.camera_names[3]}'].append(d4)
                print(f"current step : {idx} current pos : {self._get_current_pos()} target pos : {_joint}")

    def set_work_pos(self, _pos):
        sim.simxSetObjectPosition(self.client,
                                  self.work_handle,
                                  self.base_dummy,
                                  _pos,
                                  sim.simx_opmode_oneshot)
        return self._update_work_pos()

    def tip_x_changed(self, _value):
        self.ui.tip_x_label.setText(f'{_value}')
        self.work_pos[0] = _value / 1000.0
        self._update_work_pos()

    def tip_y_changed(self, _value):
        self.ui.tip_y_label.setText(f'{_value}')
        self.work_pos[1] = _value / 1000.0
        self._update_work_pos()

    def tip_z_changed(self, _value):
        self.ui.tip_z_label.setText(f'{_value}')
        self.work_pos[2] = _value / 1000.0
        self._update_work_pos()

    def _update_pos(self):
        def rad2deg(_rad):
            return _rad / math.pi * 180.0

        if self.client is not None:
            _, j1 = sim.simxGetJointPosition(self.client,
                                             self.axis1_handle,
                                             sim.simx_opmode_blocking)
            _, j2 = sim.simxGetJointPosition(self.client,
                                             self.axis2_handle,
                                             sim.simx_opmode_blocking)
            _, j3 = sim.simxGetJointPosition(self.client,
                                             self.axis3_handle,
                                             sim.simx_opmode_blocking)
            _, j4 = sim.simxGetJointPosition(self.client,
                                             self.axis4_handle,
                                             sim.simx_opmode_blocking)
            _, tip_xyz = sim.simxGetObjectPosition(self.client,
                                                   self.tip_handle,
                                                   self.grid_dummy,
                                                   sim.simx_opmode_blocking)
            _, target_xyz = sim.simxGetObjectPosition(self.client,
                                                      self.target_dummy,
                                                      self.grid_dummy,
                                                      sim.simx_opmode_blocking)
            self.ui.axis_1_label.setText(f"{round(rad2deg(j1), 2)}")
            self.ui.axis_2_label.setText(f"{round(rad2deg(j2), 2)}")
            self.ui.axis_3_label.setText(f"{round(rad2deg(j3), 2)}")
            self.ui.axis_4_label.setText(f"{round(rad2deg(j4), 2)}")
            self.ui.tip_x_label.setText(f"{round(tip_xyz[0] * 1000, 2)}")
            self.ui.tip_y_label.setText(f"{round(tip_xyz[1] * 1000, 2)}")
            self.ui.tip_z_label.setText(f"{round(tip_xyz[2] * 1000, 2)}")
            self.ui.move_tox.setText(f"{round(target_xyz[0] * 1000, 2)}")
            self.ui.move_toy.setText(f"{round(target_xyz[1] * 1000, 2)}")
            self.ui.move_toz.setText(f"{round(target_xyz[2] * 1000, 2)}")
            self.ui.count.setText(f"{self.count}")
            self._get_distance()

    def _get_distance(self):
        _, target_pos = sim.simxGetObjectPosition(self.client, self.target_dummy, -1, sim.simx_opmode_blocking)
        _, tip_pos = sim.simxGetObjectPosition(self.client, self.tip_handle, -1, sim.simx_opmode_blocking)
        distance = np.linalg.norm(np.array(target_pos) - np.array(tip_pos))
        self.delta = float(self.ui.delta.text())
        self.ui.distance.setText(f"{round(distance, 3)}")
        self.ui.s_count.setText(f"{distance // self.delta}")
        self.ui.move_to_count.setText(f"{int(distance // self.delta)}")

    def slide_x_update(self):
        _x = round(float(self.ui.tip_x_label.text()), 2)
        self.ui.tip_x.setValue(_x)

    def slide_y_update(self):
        _y = round(float(self.ui.tip_y_label.text()), 2)
        self.ui.tip_y.setValue(_y)

    def slide_z_update(self):
        _z = round(float(self.ui.tip_z_label.text()), 2)
        self.ui.tip_x.setValue(_z)

    def _update_target_pos(self):
        sim.simxSetObjectPosition(self.client,
                                  self.target_dummy,
                                  self.grid_dummy,
                                  self.target_pos,
                                  sim.simx_opmode_blocking)

    def _update_work_pos(self):
        _, _pos = sim.simxGetObjectPosition(self.client,
                                            self.work_handle,
                                            self.base_dummy,
                                            sim.simx_opmode_blocking)
        _pos = np.array(_pos) * 1000
        _joints = ex.ik(*_pos)
        self._set_pos(*_joints)
        return _joints

    def _get_current_pos(self):
        _, _q1 = sim.simxGetJointPosition(self.client, self.axis1_handle, sim.simx_opmode_blocking)
        _, _q2 = sim.simxGetJointPosition(self.client, self.axis2_handle, sim.simx_opmode_blocking)
        _, _q3 = sim.simxGetJointPosition(self.client, self.axis3_handle, sim.simx_opmode_blocking)
        _, _q4 = sim.simxGetJointPosition(self.client, self.axis4_handle, sim.simx_opmode_blocking)
        return np.array([_q1, _q2, _q3, _q4])

    def _set_pos(self, _q1, _q2, _q3, _q4):
        _q1 = sim.simxSetJointTargetPosition(self.client, self.axis1_handle, _q1, sim.simx_opmode_oneshot)
        _q2 = sim.simxSetJointTargetPosition(self.client, self.axis2_handle, _q2, sim.simx_opmode_oneshot)
        _q3 = sim.simxSetJointTargetPosition(self.client, self.axis3_handle, _q3, sim.simx_opmode_oneshot)
        _q4 = sim.simxSetJointTargetPosition(self.client, self.axis4_handle, _q4, sim.simx_opmode_oneshot)

    def target_x_changed(self, _value):
        self.ui.dummy_x_label.setText(f'{_value}')
        self.ui.dummy_x.setValue(int(float(_value)))
        self.target_pos[0] = float(_value) / 1000.0
        self._update_target_pos()

    def target_y_changed(self, _value):
        self.ui.dummy_y_label.setText(f'{_value}')
        self.ui.dummy_y.setValue(int(float(_value)))
        self.target_pos[1] = float(_value) / 1000.0
        self._update_target_pos()

    def target_z_changed(self, _value):
        self.ui.dummy_z_label.setText(f'{_value}')
        self.ui.dummy_z.setValue(int(float(_value)))
        self.target_pos[2] = float(_value) / 1000.0
        self._update_target_pos()

    def connect_sim_func(self):
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.client = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        self.ui.status.setText("simulation connect successful!!")
        self._env_set_up()

    def _save_h5py(self):
        if self.is_savable:

            # TODO
            t0 = time.time()
            max_timesteps = len(self.joint_traj)
            print("max_timesteps", max_timesteps)
            # dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
            with h5py.File(f"episode_{self.count}" + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in self.camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 512, 512, 3), dtype='uint8',
                                             chunks=(1, 512, 512, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, 4))
                # qvel = obs.create_dataset('qvel', (max_timesteps, 4))
                action = root.create_dataset('action', (max_timesteps, 4))

                for name, array in self.data_dict.items():
                    root[name][...] = array
            self.ui.status.setText(f"save episode_{self.count}" + ".hdf5 successful!!")
            self.is_savable = False

    def _env_set_up(self):
        _, self.axis1_handle = sim.simxGetObjectHandle(self.client, "joint1", sim.simx_opmode_blocking)
        _, self.axis2_handle = sim.simxGetObjectHandle(self.client, "joint2", sim.simx_opmode_blocking)
        _, self.axis3_handle = sim.simxGetObjectHandle(self.client, "joint3", sim.simx_opmode_blocking)
        _, self.axis4_handle = sim.simxGetObjectHandle(self.client, "joint4", sim.simx_opmode_blocking)
        _, self.tip_handle = sim.simxGetObjectHandle(self.client, "tip", sim.simx_opmode_blocking)
        _, self.work_handle = sim.simxGetObjectHandle(self.client, "work", sim.simx_opmode_blocking)
        _, self.grid_dummy = sim.simxGetObjectHandle(self.client, "grid_dummy", sim.simx_opmode_blocking)
        _, self.target_dummy = sim.simxGetObjectHandle(self.client, "target_dummy", sim.simx_opmode_blocking)
        _, self.base_dummy = sim.simxGetObjectHandle(self.client, "base", sim.simx_opmode_blocking)
        _, self.top_camera = sim.simxGetObjectHandle(self.client, "camera_top", sim.simx_opmode_blocking)
        _, self.focus_camera = sim.simxGetObjectHandle(self.client, "camera_focus", sim.simx_opmode_blocking)
        _, self.left_camera = sim.simxGetObjectHandle(self.client, "camera_left", sim.simx_opmode_blocking)
        _, self.right_camera = sim.simxGetObjectHandle(self.client, "camera_right", sim.simx_opmode_blocking)
        _, self.ttop_camera = sim.simxGetObjectHandle(self.client, "camera_ttop", sim.simx_opmode_blocking)
        _, self.graph = sim.simxGetObjectHandle(self.client, "graph", sim.simx_opmode_blocking)
        _, pos = sim.simxGetObjectPosition(self.client, self.work_handle, self.base_dummy, sim.simx_opmode_blocking)
        self.work_pos = np.array(pos)

    def start_sim_func(self):
        sim.simxStartSimulation(self.client, sim.simx_opmode_blocking)
        self.ui.status.setText("simulation start!")

    def pause_sim_func(self):
        sim.simxPauseSimulation(self.client, sim.simx_opmode_blocking)
        self.ui.status.setText("simulation pause")

    def stop_sim_func(self):
        sim.simxStopSimulation(self.client, sim.simx_opmode_blocking)
        self.ui.status.setText("simulation stop")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = InterFace()
    w.show()
    sys.exit(app.exec_())

# for ii in range(1, 4):
#     add_rock_objects(clientID,
#                      "block",
#                      ii,
#                      4,
#                      fr"C:\Users\85772\Desktop\rock\auto_rock_{ii}_",
#                      _pos=[0, 0, 0.4],
#                      _orient=[0, 0.1 * (ii-1), 0],
#                      _color=[random.random(), random.random(), random.random()])
#     time.sleep(2)
# sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
