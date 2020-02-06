#!/usr/bin/env python
from scenarios import HiResBallScenario
import pybullet as pb
import time
import torch as pt
import numpy as np
import cv2

np.random.seed(0)
pt.manual_seed(0)

scenario = HiResBallScenario(ball_radius=0.235, # baseball
                                robot_init_range=(-2.0, 2.0), # (-1, 1),
                                ball_x_vel_range=(-4.5, -4.5), # (-5, -3),
                                ball_init_x=8,
                                ball_y_vel=7.85,
                                ball_init_y=1,
                                camera_height=1,
                                camera_angle=np.pi/6,
                                mode=pb.GUI,
                                dt=1.0/15.0,
                                brick_texture='brick2.jpg',
                                ball_color=(0, 1, 0),
                                thirdperson=True)

state = scenario.sample_initial_dist().reshape((-1, 1))

states = [state]

writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (64, 64))

for t in range(100):
    frame = scenario.sensor(state, 0).numpy().reshape((3, 1024, 1024)).transpose([1, 2, 0])
    state = scenario.dynamics(state, 0, 0).detach().reshape((-1, 1))
    writer.write(cv2.cvtColor(np.uint8(frame * 255), cv2.COLOR_RGB2BGR))
    states.append(state)

    debug_camera_params = pb.getDebugVisualizerCamera()

    image = pb.getCameraImage(debug_camera_params[0], debug_camera_params[1], debug_camera_params[2], debug_camera_params[3])

    np.savez('ball_cam_params', width=debug_camera_params[0], height=debug_camera_params[1], view_mat=debug_camera_params[2], proj_mat=debug_camera_params[3])

    print(image[2])
    cv2.imwrite(f'ball_imgs/tpv_{t}.png', cv2.cvtColor(np.uint8(image[2]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'ball_imgs/fpv_{t}.png', cv2.cvtColor(np.uint8(frame * 255), cv2.COLOR_RGB2BGR))

    if state[2] < 0:
        states = pt.cat(states, axis=1)
        print(states[1, :])
        break

    time.sleep(1)


writer.release()
