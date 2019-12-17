from scenarios import BallScenario
import pybullet as pb
import time
import numpy as np

scenario = BallScenario(ball_radius=0.235, # baseball
                        robot_init_cov=0.001,
                        ball_x_vel_cov=0.1,
                        ball_init_x=8,
                        ball_x_vel_mean=-4.00,
                        ball_y_vel=7.85,
                        ball_init_y=1,
                        camera_height=1,
                        camera_angle=np.pi/6,
                        mode=pb.GUI)

state = scenario.sample_initial_dist()

for t in range(100):
    scenario.sensor(state, 0)
    print(state)
    state = scenario.dynamics(state, 0, 0).detach()

    if state[2] < 0:
        break
