# Make sure to have the server side running in CoppeliaSim: 
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end
import math
import random
import matplotlib.pyplot as plt
# from nvp import lvp
import numpy as np
import cv2 as cv
import sim
import time
import os
import h5py

state_list = []
JOINT_MODEL_IK = 1
JOINT_MODEL_FORCE = 0
camera_names = ['top', 'focus']

def swing_generate(_pt):
    _x, _y, _z = _pt
    return _x, _y + 0.06, _z - 0.05

# record hdf5
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
        if self.velocity_plan:
            self._lvp()

    def _lvp(self):
        self.lvp_list.append(self.path[0])
        for _i in range(1, len(self.path)):
            v = lvp(0, 0, self.path[_i - 1], self.path[_i], 8)
            plt.plot(v)
            plt.show()
            for _v in v:
                self.lvp_list.append(np.array(self.lvp_list[-1]) + _v * 0.002)

    def __next__(self):
        if self.velocity_plan:
            self._idx += 1
            if self._idx < len(self.lvp_list):
                return self.lvp_list[self._idx]
            else:
                raise StopIteration
        else:
            self._idx += 1
            if self._idx < len(self.interp) - 1:
                return self.interp[self._idx]
            else:
                raise StopIteration

    def __len__(self):
        return len(self.interp)


class EnvState:
    def __init__(self, _t, qs, __pos, quat, cameras):
        self.tt = _t
        self.q1 = qs[0]
        self.q2 = qs[1]
        self.q3 = qs[2]
        self.q4 = qs[3]
        self.top = cameras[0]
        self.focus = cameras[1]
        self.pos = __pos
        self.quat = quat

    def qs(self):
        return np.array([self.q1, self.q2, self.q3, self.q4])

    def get_action(self):
        return np.array(self.pos + self.quat)

    def get_top(self):
        return self.top

    def get_focus(self):
        return self.focus


def get_current_image(_camera_top, _camera_focus):
    _, ret, _img_top = sim.simxGetVisionSensorImage(clientID, _camera_top, 0, sim.simx_opmode_blocking)
    _, ret, _img_focus = sim.simxGetVisionSensorImage(clientID, _camera_focus, 0, sim.simx_opmode_blocking)
    tt = time.time()
    _img_top = np.array(_img_top, dtype=np.uint8)
    _img_top.resize([512, 512, 3])
    _img_focus = np.array(_img_focus, dtype=np.uint8)
    _img_focus.resize([512, 512, 3])
    print(f"time: {time.time() - tt}")
    return _img_top, _img_focus


def get_current_pos(_q1, _q2, _q3, _q4):
    _, pos1 = sim.simxGetJointPosition(clientID, _q1, sim.simx_opmode_blocking)
    _, pos2 = sim.simxGetJointPosition(clientID, _q2, sim.simx_opmode_blocking)
    _, pos3 = sim.simxGetJointPosition(clientID, _q3, sim.simx_opmode_blocking)
    _, pos4 = sim.simxGetJointPosition(clientID, _q4, sim.simx_opmode_blocking)
    return [pos1 * 180 / np.pi, pos2 * 180 / np.pi, pos3 * 180 / np.pi, pos4 *  180 / np.pi]


def get_current_pos_rad(_q1, _q2, _q3, _q4):
    _, pos1 = sim.simxGetJointPosition(clientID, _q1, sim.simx_opmode_blocking)
    _, pos2 = sim.simxGetJointPosition(clientID, _q2, sim.simx_opmode_blocking)
    _, pos3 = sim.simxGetJointPosition(clientID, _q3, sim.simx_opmode_blocking)
    _, pos4 = sim.simxGetJointPosition(clientID, _q4, sim.simx_opmode_blocking)
    return [pos1, pos2, pos3, pos4]


def get_current_state(_q1, _q2, _q3, _q4, __pos, __quat, _camera_top, _camera_focus, _start):
    __qs = get_current_pos(_q1, _q2, _q3, _q4)
    _images = get_current_image(_camera_top, _camera_focus)
    return EnvState(time.time() - _start, __qs, __pos, __quat, _images)


def get_ee_handles(clientID):
    _, _obj = sim.simxGetObjectHandle(clientID, "work", sim.simx_opmode_blocking)
    _, _tip = sim.simxGetObjectHandle(clientID, "tip", sim.simx_opmode_blocking)
    _, _camera_top = sim.simxGetObjectHandle(clientID, "camera_top", sim.simx_opmode_blocking)
    _, _camera_focus = sim.simxGetObjectHandle(clientID, "camera_focus", sim.simx_opmode_blocking)
    _, _home = sim.simxGetObjectHandle(clientID, "home", sim.simx_opmode_blocking)
    _, _grid = sim.simxGetObjectHandle(clientID, "grid", sim.simx_opmode_blocking)
    _, _q1 = sim.simxGetObjectHandle(clientID, "joint1", sim.simx_opmode_blocking)
    _, _q2 = sim.simxGetObjectHandle(clientID, "joint2", sim.simx_opmode_blocking)
    _, _q3 = sim.simxGetObjectHandle(clientID, "joint3", sim.simx_opmode_blocking)
    _, _q4 = sim.simxGetObjectHandle(clientID, "joint4", sim.simx_opmode_blocking)
    return [_obj, _tip, _camera_top, _camera_focus, _home, _grid, _q1, _q2, _q3, _q4]


def get_handles(clientID):
    _, _camera_top = sim.simxGetObjectHandle(clientID, "camera_top", sim.simx_opmode_blocking)
    _, _camera_focus = sim.simxGetObjectHandle(clientID, "camera_focus", sim.simx_opmode_blocking)
    _, _q1 = sim.simxGetObjectHandle(clientID, "joint1", sim.simx_opmode_blocking)
    _, _q2 = sim.simxGetObjectHandle(clientID, "joint2", sim.simx_opmode_blocking)
    _, _q3 = sim.simxGetObjectHandle(clientID, "joint3", sim.simx_opmode_blocking)
    _, _q4 = sim.simxGetObjectHandle(clientID, "joint4", sim.simx_opmode_blocking)
    _, _grid = sim.simxGetObjectHandle(clientID, "grid", sim.simx_opmode_blocking)
    return [_camera_top, _camera_focus, _q1, _q2, _q3, _q4, _grid]


print('Program started')
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
names = "block"

print('Connected to remote API server')


def upup(_pts):
    return [_pts[0], _pts[1], _pts[2] + 0.1]


for _ii in range(50):
    sim.simxLoadScene(clientID,
                      "E:\\doctor\\teaching_graps\\act-main\\similate_scence_ee.ttt",
                      0,
                      sim.simx_opmode_blocking)
    obj, tip, camera_top, camera_focus, home, grid, q1, q2, q3, q4 = get_ee_handles(clientID)
    print(q1, q2, q3, q4, grid)
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    # _ = sim.simxCallScriptFunction(clientID,
    #                                "remoteApiCommandServer",
    #                                sim.sim_scripttype_childscript,
    #                                'setJointModel',
    #                                [JOINT_MODEL_IK],
    #                                [1.0],
    #                                ["ss"],
    #                                bytearray(),
    #                                sim.simx_opmode_blocking)
    data_dict = {'/observations/qpos': [],
                 '/action': []}

    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # 自动化随机生成
    object_orient = [0.0, 0.0, 0.0]
    xy_list = []
    color_list = []
    for ii in range(4):
        object_color = [random.random(), random.random(), random.random()]
        _x = _y = 0
        while True:
            _x = random.randint(-1, 2)
            _y = random.randint(-2, 2)
            if [_x, _y] in xy_list:
                continue
            else:
                xy_list.append([_x, _y])
                color_list.append(object_color)
                break
    for ii, items in enumerate(xy_list):
        _ = sim.simxCallScriptFunction(clientID,
                                       "remoteApiCommandServer",
                                       sim.sim_scripttype_childscript,
                                       'generateBlock',
                                       [ii],
                                       [items[0] * 0.11, items[1] * 0.11, 0.1] + object_orient + color_list[ii],
                                       [names],
                                       bytearray(),
                                       sim.simx_opmode_blocking)
        time.sleep(1)
    _, home_pos = sim.simxGetObjectPosition(clientID, home, grid, sim.simx_opmode_blocking)
    print(xy_list)
    dummy_pos_list = []
    block_list = []
    for ii in range(4):
        _, dummy_name = sim.simxGetObjectHandle(clientID, f"block_dummy{ii}", sim.simx_opmode_blocking)
        _, pos = sim.simxGetObjectPosition(clientID, dummy_name, grid, sim.simx_opmode_blocking)
        dummy_pos_list.append(pos)
        _, block_handle = sim.simxGetObjectHandle(clientID, f"block{ii}", sim.simx_opmode_blocking)
        block_list.append(block_handle)
    print("block_list", block_list)

    points = [home_pos,
              dummy_pos_list[0],
              swing_generate(dummy_pos_list[0]),
              upup(swing_generate(dummy_pos_list[0])),
              dummy_pos_list[1],
              swing_generate(dummy_pos_list[1]),
              upup(swing_generate(dummy_pos_list[1])),
              dummy_pos_list[2],
              swing_generate(dummy_pos_list[2]),
              upup(swing_generate(dummy_pos_list[2])),
              dummy_pos_list[3],
              swing_generate(dummy_pos_list[3]),
              ]
    inter = InterpPath()
    inter.set_path(points)
    start = time.time()
    # ee -----------------------------------------------------------------------部分
    joint_traj = []
    for ii, ll in enumerate(inter):
        # t_start = sim.simxGetPingTime(clientID)
        sim.simxSetObjectPosition(clientID, obj, grid, ll, sim.simx_opmode_blocking)
        for block in block_list:
            _, pos = sim.simxGetObjectPosition(clientID, block, grid, sim.simx_opmode_blocking)
            if pos[2] < 0:
                sim.simxRemoveObject(clientID, block, sim.simx_opmode_blocking)
        # img_top, img_focus = get_current_image(camera_top, camera_focus)
        t = time.time()
        qpos = get_current_pos(q1, q2, q3, q4)
        joint_traj.append(qpos)

    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    time.sleep(3)
    sim.simxLoadScene(clientID,
                      "E:\\doctor\\teaching_graps\\act-main\\similate_scence.ttt",
                      0,
                      sim.simx_opmode_blocking)
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    camera_top, camera_focus, q1, q2, q3, q4, grid = get_handles(clientID)
    print(camera_top, camera_focus, q1, q2, q3, q4, grid)
    block_list = [43, 45, 47, 49]
    # for ii in range(4):
    #     _, block_handle = sim.simxGetObjectHandle(clientID, f"block{ii}", sim.simx_opmode_blocking)
    #     block_list.append(block_handle)
    print("block_list", block_list)
    # 重新生成方块
    for ii, items in enumerate(xy_list):
        _ = sim.simxCallScriptFunction(clientID,
                                       "remoteApiCommandServer",
                                       sim.sim_scripttype_childscript,
                                       'generateBlock',
                                       [ii],
                                       [items[0] * 0.11, items[1] * 0.11, 0.1] + object_orient + color_list[ii],
                                       [names],
                                       bytearray(),
                                       sim.simx_opmode_blocking)
        time.sleep(1)
    for ii in range(len(joint_traj)):
        # t_start = sim.simxGetPingTime(clientID)
        # _, j1 = sim.simxGetObjectHandle(clientID, "UR5_joint1", sim.simx_opmode_blocking)
        # sim.simxSetJointTargetPosition(clientID, q1, ii / 180 * math.pi, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID, q1, joint_traj[ii][0] / 180 * np.pi, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID, q2, joint_traj[ii][1] / 180 * np.pi, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID, q3, joint_traj[ii][2] / 180 * np.pi, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID, q4, joint_traj[ii][3] / 180 * np.pi, sim.simx_opmode_streaming)
        for block in block_list:
            _, pos = sim.simxGetObjectPosition(clientID, block, grid, sim.simx_opmode_blocking)
            print(pos)
            if pos[2] <= 0.05:
                sim.simxRemoveObject(clientID, block, sim.simx_opmode_blocking)
        print("------------------------------------------")
        _, pos = sim.simxGetObjectPosition(clientID, tip, grid, sim.simx_opmode_blocking)
        _, quat = sim.simxGetObjectQuaternion(clientID, tip, grid, sim.simx_opmode_blocking)
        t = time.time()
        state = get_current_state(q1, q2, q3, q4, pos, quat, camera_top, camera_focus, t)
        state_list.append(state)
        data_dict['/observations/qpos'].append(state.qs())
        data_dict['/action'].append(joint_traj[ii])
        data_dict[f'/observations/images/{camera_names[0]}'].append(state.top)
        data_dict[f'/observations/images/{camera_names[1]}'].append(state.focus)
    # HDF5
    t0 = time.time()
    max_timesteps = len(joint_traj)
    print("max_timesteps", max_timesteps)
    # dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
    with h5py.File(f"episode_mm{_ii}" + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 512, 512, 3), dtype='uint8',
                                     chunks=(1, 512, 512, 3), )
        # compression='gzip',compression_opts=2,)
        # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        qpos = obs.create_dataset('qpos', (max_timesteps, 4))
        # qvel = obs.create_dataset('qvel', (max_timesteps, 4))
        action = root.create_dataset('action', (max_timesteps, 4))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs\n')
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    time.sleep(1)
print('Program ended')