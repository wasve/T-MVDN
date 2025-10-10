# -*- coding: UTF-8 -*-
"""================================================================
>> Project -> File  : act-main -> sim_control
>> IDE              : PyCharm
>> Author           : Wasve
>> Date             : 2024/9/18 19:51
>> Desc             : None
================================================================"""
import numpy as np
import cv2 as cv
import sim
import time
import os
import h5py

CAMERA_EX_NAME = "camera_top"
CAMERA_IN_NAME = "camera_focus"


class InterpPath:

    def __init__(self):
        self.path = None
        self.delta = 0.001
        self._idx = -1
        self.interp = []
        self.lvp_list = []
        self.velocity_plan = False

    def __iter__(self):
        return self

    def set_path(self, _path):
        self.path = _path
        nums = len(self.path)
        interp_list = []
        delta_list = []
        self.interp = []
        self.interp.append(self.path[0])
        for _i in range(1, nums):
            _delta = np.array(self.path[_i]) - np.array(self.path[_i - 1])
            _max_delta = max(abs(_delta))
            interp_count = int(round(_max_delta / self.delta, 0))
            interp_list.append(interp_count)
            delta_list.append(_delta / interp_count)
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
                self._idx = -1
                raise StopIteration

    def __len__(self):
        return len(self.interp)


class BrokenArm:

    def __init__(self):
        print('Program started')
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)
        # self.restart_sim()
        self.interp = InterpPath()
        self.interp.delta = 0.01
        self._set_up_sim()
        self.joint_handles = [0, 0, 0, 0]
        ...

    def _get_img(self):
        _, ret, _img_top = sim.simxGetVisionSensorImage(self.clientID, self.cam_ex_handle, 0, sim.simx_opmode_blocking)
        _, ret, _img_focus = sim.simxGetVisionSensorImage(self.clientID, self.cam_in_handle, 0,
                                                          sim.simx_opmode_blocking)
        _img_top = np.array(_img_top, dtype=np.uint8)
        _img_top.resize([512, 512, 3])
        _img_focus = np.array(_img_focus, dtype=np.uint8)
        _img_focus.resize([512, 512, 3])
        return _img_top, _img_focus

    def record_hdf5(self):
        self.start()

        def swing_generate(_pt):
            _x, _y, _z = _pt
            return _x, _y + 0.1, _z - 0.05

        _, dummy_home = sim.simxGetObjectHandle(self.clientID, "home", sim.simx_opmode_blocking)
        _, dummy_work_left_top = sim.simxGetObjectHandle(self.clientID, "work_left_top", sim.simx_opmode_blocking)
        _, dummy_work_right_top = sim.simxGetObjectHandle(self.clientID, "work_right_top", sim.simx_opmode_blocking)
        _, dummy_work_right_down = sim.simxGetObjectHandle(self.clientID, "work_right_down", sim.simx_opmode_blocking)
        _, dummy_work_left_down = sim.simxGetObjectHandle(self.clientID, "work_left_down", sim.simx_opmode_blocking)
        _, home = sim.simxGetObjectPosition(self.clientID, dummy_home, -1, sim.simx_opmode_blocking)
        _, work_left_top = sim.simxGetObjectPosition(self.clientID, dummy_work_left_top, -1, sim.simx_opmode_blocking)
        _, work_right_top = sim.simxGetObjectPosition(self.clientID, dummy_work_right_top, -1, sim.simx_opmode_blocking)
        _, work_right_down = sim.simxGetObjectPosition(self.clientID, dummy_work_right_down, -1,
                                                       sim.simx_opmode_blocking)
        _, work_left_down = sim.simxGetObjectPosition(self.clientID, dummy_work_left_down, -1, sim.simx_opmode_blocking)

        self.j1 = sim.simxGetObjectHandle(self.clientID, "joint1", sim.simx_opmode_blocking)
        self.j1 = sim.simxGetObjectHandle(self.clientID, "joint1", sim.simx_opmode_blocking)
        self.j1 = sim.simxGetObjectHandle(self.clientID, "joint1", sim.simx_opmode_blocking)
        self.j1 = sim.simxGetObjectHandle(self.clientID, "joint1", sim.simx_opmode_blocking)
        points = [home,
                  work_left_top,
                  swing_generate(work_left_top),
                  home,
                  work_right_top,
                  swing_generate(work_right_top),
                  home,
                  work_right_down,
                  swing_generate(work_right_down),
                  home,
                  work_left_down,
                  swing_generate(work_left_down),
                  home]
        self.move_to(points)

    def _set_up_sim(self):
        # Get handle to camera
        sim_ret, self.cam_ex_handle = sim.simxGetObjectHandle(self.clientID, CAMERA_EX_NAME,
                                                              sim.simx_opmode_oneshot_wait)
        sim_ret, self.cam_in_handle = sim.simxGetObjectHandle(self.clientID, CAMERA_IN_NAME,
                                                              sim.simx_opmode_oneshot_wait)

        sim_ret, self.work_dummy = sim.simxGetObjectHandle(self.clientID, "work", sim.simx_opmode_blocking)
        sim_ret, self.joint1 = sim.simxGetObjectHandle(self.clientID, "joint1", sim.simx_opmode_blocking)
        sim_ret, self.joint2 = sim.simxGetObjectHandle(self.clientID, "joint2", sim.simx_opmode_blocking)
        sim_ret, self.joint3 = sim.simxGetObjectHandle(self.clientID, "joint3", sim.simx_opmode_blocking)
        sim_ret, self.joint4 = sim.simxGetObjectHandle(self.clientID, "joint4", sim.simx_opmode_blocking)

    def move_to(self, points):
        self.interp.set_path(points)
        for _pt in self.interp:
            sim.simxSetObjectPosition(self.clientID, self.work_dummy, -1, _pt, sim.simx_opmode_blocking)
        ...

    def step(self, _qs):
        sim.simxSetJointPosition(clientID=self.clientID, )
        ...

    def initialize_episode(self):
        ...

    def get_env_state(self):
        ...

    def get_reward(self):
        ...

    def get_qpos(self):
        _, pos1 = sim.simxGetJointPosition(self.clientID, self.joint1, sim.simx_opmode_blocking)
        _, pos2 = sim.simxGetJointPosition(self.clientID, self.joint2, sim.simx_opmode_blocking)
        _, pos3 = sim.simxGetJointPosition(self.clientID, self.joint3, sim.simx_opmode_blocking)
        _, pos4 = sim.simxGetJointPosition(self.clientID, self.joint4, sim.simx_opmode_blocking)
        return pos1, pos2, pos3, pos4

    def get_observation(self):
        ...

    def __del__(self):
        sim.simxFinish(self.clientID)
        print('Program ended')

    def restart_sim(self):
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        time.sleep(1)
        # self._set_up_sim()
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)

    def start(self):
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)


if __name__ == "__main__":
    rbt = BrokenArm()
    rbt.record_hdf5()
    rbt.restart_sim()
    rbt.record_hdf5()
