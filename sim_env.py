import random

import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from math import cos, sin, atan2, sqrt, acos, pi
import math
from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

import numpy as np
import cv2 as cv
import sim
import time
import os
import h5py

TOP = "camera_top"
LEFT = "camera_left"
TTOP = "camera_ttop"
FOCUS = "camera_focus"

def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_ai_broken' in task_name:
        env = BrokenArmEnvironment()
    else:
        raise NotImplementedError
    return env

def add_rock_objects(client, _name, _num, _part_num, _path, _pos, _orient, _color, _scale):
    _ = sim.simxCallScriptFunction(client,
                                   "server",
                                   sim.sim_scriptt,
                                   'addRock',
                                   [_num, _part_num],
                                   _pos + _orient + _color + [_scale],
                                   [_name, _path],
                                   bytearray(),
                                   sim.simx_opmode_blocking)

class BrokenExcavator:

    def __init__(self):
        self.d1 = 126
        self.a1 = -70.5
        self.alpha1 = pi / 2

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

        # 计算总的变换矩阵
        T = T1 @ T2 @ T3 @ T4

        # 提取 px, py, pz 并应用 35 的偏移量
        px = T[0, 3] + 35 * math.cos(theta1)
        py = T[1, 3] + 35 * math.sin(theta1)
        pz = T[2, 3]
        return round(px / 1000.0, 4), round(py / 1000.0, 4), round(pz / 1000.0, 4)


class BrokenArmEnvironment:

    def __init__(self):
        print('Program started')
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
        # self.restart_sim()
        self.cube_handles = []
        self.step_num = 0
        self._set_up_sim()
        self.joint_handles = [0, 0, 0, 0]
        self.rock_num = 3
        self.max_reward = 12
        self.cube_info_list = []
        self.observation = {}
        self.reward = 0
        self.ex = BrokenExcavator()

        ...

    def _go_edge(self):
        sim.simxSetJointTargetPosition(self.clientID, self.joint1, -2.02 / 180 * np.pi, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.joint2, 14.37 / 180 * np.pi, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.joint3, -25.26 / 180 * np.pi, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.joint4, 10.89 / 180 * np.pi, sim.simx_opmode_oneshot)


    def reset(self):
        # self.stop_simulation()
        # time.sleep(1)
        # self.start_simulation()
        height = 0.4
        # height = 0.2
        self.cube_info_list = []
        time.sleep(2)


            # for ii in range(self.rock_num):
            #     add_rock_objects(self.clientID,
            #                      "block",
            #                      ii,
            #                      4,
            #                      fr"C:\Users\85772\Desktop\rock\auto_rock_{random.randint(1, 10)}_",
            #                      _pos=[0.1, 0.0, height],
            #                      _orient=[0.0, 0.0, 0.0],
            #                      _color=[152 / 255, 160 / 255, 169 / 255],
            #                      _scale=random.uniform(0.04, 0.045))
            #     time.sleep(2)


        self._set_up_sim()
        self.go_home()
        time.sleep(1)
        self.__update_observation()

    def set_actions_path(self, float_packs):
        packed_data = sim.simxPackFloats(float_packs)
        sim.simxSetStringSignal(self.clientID, 'coordinates_signal', packed_data, sim.simx_opmode_blocking)

    def __update_observation(self):
        _qs = self.get_qpos()
        _imgs = self._get_img()
        self.observation = {"images": {"top": np.array(_imgs[0]),
                                       "ttop": np.array(_imgs[1]),
                                       "left": np.array(_imgs[2]),
                                       "focus": np.array(_imgs[3])},
                            "qpos": np.array([_qs[0], _qs[1], _qs[2], _qs[3]])}
        self.check_reward()
        print(f"step{self.step_num}", _qs, f"reward={self.reward}")
        ...

    def go_home(self):
        sim.simxSetJointTargetPosition(self.clientID, self.joint1, -2.02 / 180 * np.pi, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.joint2, 14.37 / 180 * np.pi, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.joint3, -25.26 / 180 * np.pi, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.joint4, 10.89 / 180 * np.pi, sim.simx_opmode_oneshot)

    def _get_img(self):
        _, ret, _img_top = sim.simxGetVisionSensorImage(self.clientID, self.cam_top_handle,  0, sim.simx_opmode_blocking)
        _, ret, _img_focus = sim.simxGetVisionSensorImage(self.clientID, self.cam_focus_handle, 0, sim.simx_opmode_blocking)
        _, ret, _img_left = sim.simxGetVisionSensorImage(self.clientID, self.cam_left_handle, 0, sim.simx_opmode_blocking)
        # _, ret, _img_right = sim.simxGetVisionSensorImage(self.client, self.right_camera, 0, sim.simx_opmode_blocking)
        _, ret, _img_ttop = sim.simxGetVisionSensorImage(self.clientID, self.cam_ttop_handle, 0, sim.simx_opmode_blocking)
        _img_top = np.array(_img_top).astype(np.uint8)
        _img_top.resize([512, 512, 3])
        _img_focus = np.array(_img_focus).astype(np.uint8)
        _img_focus.resize([512, 512, 3])
        _img_left = np.array(_img_left).astype(np.uint8)
        _img_left.resize([512, 512, 3])
        # _img_right = np.array(_img_right).astype(np.uint8)
        # _img_right.resize([512, 512, 3])
        _img_ttop = np.array(_img_ttop).astype(np.uint8)
        _img_ttop.resize([512, 512, 3])
        return _img_top, _img_ttop, _img_left, _img_focus

    def _set_up_sim(self):
        # Get handle to camera
        sim_ret, self.cam_top_handle = sim.simxGetObjectHandle(self.clientID, TOP, sim.simx_opmode_blocking)
        sim_ret, self.cam_ttop_handle = sim.simxGetObjectHandle(self.clientID, TTOP, sim.simx_opmode_blocking)
        sim_ret, self.cam_left_handle = sim.simxGetObjectHandle(self.clientID, LEFT, sim.simx_opmode_blocking)
        sim_ret, self.cam_focus_handle = sim.simxGetObjectHandle(self.clientID, FOCUS, sim.simx_opmode_blocking)
        sim_ret, self.work_dummy = sim.simxGetObjectHandle(self.clientID, "work", sim.simx_opmode_blocking)
        sim_ret, self.joint1 = sim.simxGetObjectHandle(self.clientID, "joint1", sim.simx_opmode_blocking)
        sim_ret, self.joint2 = sim.simxGetObjectHandle(self.clientID, "joint2", sim.simx_opmode_blocking)
        sim_ret, self.joint3 = sim.simxGetObjectHandle(self.clientID, "joint3", sim.simx_opmode_blocking)
        sim_ret, self.joint4 = sim.simxGetObjectHandle(self.clientID, "joint4", sim.simx_opmode_blocking)
        sim_ret, self.grid = sim.simxGetObjectHandle(self.clientID, "grid", sim.simx_opmode_blocking)
        self.step_num = 0
        # Get camera pose and intrinsics in simulation
        # sim_ret, cam_position = sim.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, sim.simx_opmode_blocking)
        # sim_ret, cam_orientation = sim.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, sim.simx_opmode_blocking)
        # cam_trans = np.eye(4,4)#对角线形状的onehot编码10000100
        # cam_trans[0:3,3] = np.asarray(cam_position)
        # cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        # cam_rotm = np.eye(4,4)
        # cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))#（矩阵求逆（欧拉角转为旋转矩阵））
        # self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose#矩阵乘积
        # self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        # self.cam_depth_scale = 1
        #
        # # Get background image
        # self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        # self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def check_reward(self):
        self.reward = 0
        # for i in range(self.rock_num):
        #     for j in range(4):
        #         res = sim.simxGetObjectHandle(self.clientID, f"rock_{i}_{j}", sim.simx_opmode_blocking)
        #         if res[0] != 0:
        #             self.reward += 1

    def step(self, _qs):
        sim.simxSetJointTargetPosition(self.clientID, self.joint1, _qs[0], sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.joint2, _qs[1], sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.joint3, _qs[2], sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.joint4, _qs[3], sim.simx_opmode_oneshot)
        self.step_num += 1
        self.__update_observation()

    def __sample_cube_pose(self):
        # 弃用
        """
        生成方框的位置与颜色
        """
        xy_list = []
        color_list = []
        pos_list = []
        for ii in range(self.cube_num):
            object_color = [random.random(), random.random(), random.random()]
            _x = _y = 0
            while True:
                _x = random.randint(-1, 2)
                _y = random.randint(-2, 2)
                if [_x, _y] in xy_list:
                    continue
                else:
                    xy_list.append([_x, _y])
                    pos_list.append([0.11 * _x, 0.11 * _y, 0.1])
                    color_list.append(object_color)
                    break
        for _ii in range(self.cube_num):
            _pos = pos_list[_ii]
            _color = color_list[_ii]
            _info = _pos + [0.0, 0.0, 0.0] + _color
            self.cube_info_list.append(_info)
        return self.cube_info_list

    def get_reward(self):
        ...

    def get_qpos(self):
        _, pos1 = sim.simxGetJointPosition(self.clientID, self.joint1, sim.simx_opmode_blocking)
        _, pos2 = sim.simxGetJointPosition(self.clientID, self.joint2, sim.simx_opmode_blocking)
        _, pos3 = sim.simxGetJointPosition(self.clientID, self.joint3, sim.simx_opmode_blocking)
        _, pos4 = sim.simxGetJointPosition(self.clientID, self.joint4, sim.simx_opmode_blocking)
        return pos1, pos2, pos3, pos4

    def __del__(self):
        sim.simxFinish(self.clientID)
        print('Program ended')

    def start_simulation(self):
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)

    def stop_simulation(self):
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

