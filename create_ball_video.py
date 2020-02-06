#!/usr/bin/env python
import policies
import scenarios
from networks import PiNetTV, QNetTV, PiNetShared, QNetShared
import torch as pt
import torch.nn as nn
import numpy as np
import sys
import pybullet as pb
import matplotlib.pyplot as plt
import cv2
import shutil

from policies import rollout

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

from pathlib import Path

batch_size = 1
ntrvs = 8

textures = [
    'brick_backgrounds/training.jpg',
    'brick_backgrounds/test1.jpg',
    'brick_backgrounds/test2.jpg',
    'brick_backgrounds/test3.jpg',
    'brick_backgrounds/test4.jpg',
    'brick_backgrounds/test5.jpg',
    'brick_backgrounds/test6.jpg',
    'brick_backgrounds/test7.jpg',
]

policies = [
    'models/good_initialization',
    'models/Ball_tradeoff_20_epoch_1_mi_6.203'
]

np.random.seed(0)
pt.manual_seed(0)

scenario = scenarios.BallScenario(ball_radius=0.235, # baseball
                                robot_init_range=(-2.0, 2.0), # (-1, 1),
                                ball_x_vel_range=(-4.5, -4.5), # (-5, -3),
                                ball_init_x=8,
                                ball_y_vel=7.85,
                                ball_init_y=1,
                                camera_height=1,
                                camera_angle=np.pi/6,
                                mode=pb.GUI,
                                dt=1.0/15.0,
                                ball_color=(0, 1, 0),
                                thirdperson=True)

def make_preprocess_net():
    return nn.Sequential(
        nn.Conv2d(3, 6, 4, stride=2),
        nn.ELU(),
        nn.Conv2d(6, 6, 4, stride=2),
    )

test_net = make_preprocess_net()
net_out_size = test_net(pt.rand((1, 3, 64, 64))).numel()

def make_pi_sequence(t: int):
    return nn.Sequential(
        nn.Linear(ntrvs, scenario.ninputs*2)

    )

def make_q_sequence(t: int):
    return nn.Sequential(
        nn.Linear(net_out_size + ntrvs, 32),
        nn.Tanh(),
        nn.Linear(32, ntrvs*2),
        nn.Tanh()
    )

camera_params = np.load('ball_cam_params.npz')

for j in range(len(policies)):
    np.random.seed(0)
    pt.manual_seed(0)

    pi_net = PiNetShared(make_pi_sequence)
    q_net = QNetShared(make_q_sequence, make_preprocess_net, reshape_to=scenario.image_shape)

    loaded_models = pt.load(policies[j])
    pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
    q_net.load_state_dict(loaded_models['q_net_state_dict'])

    for texture in textures:
        pb.resetSimulation()
        scenario = scenarios.BallScenario(ball_radius=0.235, # baseball
                                        robot_init_range=(-2.0, 2.0), # (-1, 1),
                                        ball_x_vel_range=(-4.5, -4.5), # (-5, -3),
                                        ball_init_x=8,
                                        ball_y_vel=7.85,
                                        ball_init_y=1,
                                        camera_height=1,
                                        camera_angle=np.pi/6,
                                        mode=pb.DIRECT,
                                        dt=1.0/15.0,
                                        brick_texture=texture,
                                        ball_color=(0, 1, 0),
                                        thirdperson=True)

        states, outputs, samples, trvs, inputs, costs = rollout(pi_net, q_net, ntrvs, scenario, scenario.horizon, batch_size, pt.device('cpu'))

        dir_path = f'ball_videos/policy{j}/{texture.split("/")[-1].split(".")[0]}'
        shutil.rmtree(dir_path, ignore_errors=True)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        for i in range(batch_size):
            writer = cv2.VideoWriter(f'{dir_path}/{i}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (camera_params['width'], camera_params['height']))

            for t in range(scenario.horizon + 1):
                scenario.video_sensor(states[:, t, i], 0)

                #debug_camera_params = pb.getDebugVisualizerCamera()

                image = pb.getCameraImage(camera_params['width'], camera_params['height'], camera_params['view_mat'], camera_params['proj_mat'])
                writer.write(cv2.cvtColor(np.uint8(image[2]), cv2.COLOR_RGB2BGR))
                print(f'{i}.{t}')

            writer.release()
