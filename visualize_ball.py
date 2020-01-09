#!/usr/bin/env python
from scenarios import BallScenario
import pybullet as pb
import time
import torch as pt
import numpy as np
import cv2

scenario = BallScenario(ball_radius=0.235, # baseball
                        robot_init_range=(-1, 1),
                        ball_x_vel_range=(-5, -3),
                        ball_init_x=8,
                        ball_y_vel=7.85,
                        ball_init_y=1,
                        camera_height=1,
                        camera_angle=np.pi/6,
                        mode=pb.DIRECT, dt=1.0/15.0)

state = scenario.sample_initial_dist().reshape((-1, 1))

states = [state]

writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (64, 64))

for t in range(100):
    frame = scenario.sensor(state, 0).numpy().reshape((3, 64, 64)).transpose([1, 2, 0])
    state = scenario.dynamics(state, 0, 0).detach().reshape((-1, 1))
    writer.write(cv2.cvtColor(np.uint8(frame * 255), cv2.COLOR_RGB2BGR))
    states.append(state)

    if t == 20:
        cv2.imwrite('sample_image.png', cv2.cvtColor(np.uint8(frame * 255), cv2.COLOR_RGB2BGR))

    if state[2] < 0:
        states = pt.cat(states, axis=1)
        print(states[1, :])
        break

writer.release()
