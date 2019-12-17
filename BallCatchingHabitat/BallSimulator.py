# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import random
import subprocess
import magnum as mn
import numpy as np
from PIL import Image

import habitat_sim
import habitat_sim.utils.common as ut

# --------------------------
# Utilities
# --------------------------


def save_observation(observations, frame, envNum, prefix=""):
    if "color" in observations:
        rgb_img = Image.fromarray(observations["color"], mode="RGBA")
        rgb_img.save("CapturedFrames/Environment" + str(envNum) +"/" + prefix + "rgba.%05d.png" % frame)
    if "depth" in observations:
        depth_img = Image.fromarray(
            (observations["depth"] / 10 * 255).astype(np.uint8), mode="L"
        )
        depth_img.save(
            "CapturedFrames/Environment" + str(envNum) + "/" + prefix + "depth.%05d.png" % frame
        )

def RunSimulation(envNum, record, path, Coords, seed): 
    
    settings = {"image_height": 720, "image_width": 1080, "random_seed": seed}
    # pick a scene file to load
    scene_file = path 

    if record:
        os.system("mkdir CapturedFrames")
        os.system(("mkdir CapturedFrames/Environment" + str(envNum)))

    # --------------------------
    # Simulator Setup
    # --------------------------

    # create a SimulatorConfiguration object
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene.id = scene_file
    # configure the simulator to intialize a PhysicsManager
    sim_cfg.enable_physics = True
    # configure the simulator to load a specific simulator configuration file (define paths you your object models here)
    sim_cfg.physics_config_file = "data/default.phys_scene_config.json"
    # pick this based on your device
    relative_camera_position = [0.0, 0, 0.0]
    # Note: all sensors must have the same resolution
    sensors = {
        "color": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["image_height"], settings["image_width"]],
            "position": relative_camera_position,
        },
        "depth": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["image_height"], settings["image_width"]],
            "position": relative_camera_position,
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.SensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]

        sensor_specs.append(sensor_spec)

    # setup the agent/camera
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    
    agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(amount=0.2)
        ),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }                                                                   

    # generate the full simulator configuration
    combined_cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    # initialize the simulator
    sim = habitat_sim.Simulator(combined_cfg)
    sim.agents[0].controls.move_filter_fn = sim._scene_collision_step_filter 

    # set random seed
    random.seed(settings["random_seed"])
    sim.seed(settings["random_seed"])

    # initialize the agent state
    sim.initialize_agent(0)
    # ----------------------------------
    # Quadrotor Prototype Functionality
    # ----------------------------------

    # assuming the first entry in the object list corresponds to the quadrotor model
    quadrotor_id = sim.add_object(0)
    
    # the object will not be affected by forces such as gravity and will remain static until explicitly moved
    sim.set_object_motion_type(
            habitat_sim.physics.MotionType.KINEMATIC, quadrotor_id
        )

    ## Randomly initialize the Ball

    initial_height = 0.00001
    possible_initial_point = sim.pathfinder.get_random_navigable_point() + np.array([0, initial_height, 0])
    sim.set_translation(possible_initial_point, quadrotor_id)
    while(sim.contact_test(quadrotor_id)):
        possible_initial_point = sim.pathfinder.get_random_navigable_point() + np.array([0, initial_height, 0])
        sim.set_translation(possible_initial_point, quadrotor_id)

    # place the object in the air
    # sim.set_translation(np.array([-0.569043, 2.04804, 13.6156]), quadrotor_id)
    sim.agents[0].scene_node.translation = sim.get_translation(quadrotor_id)

    # static_3rd_person_camera_position = np.array([-0.569043, 2.04804, 12.6156])
    static_3rd_person_camera_position = np.array(sim.get_translation(quadrotor_id)) - np.array([0, 0, 2])

    initTranslation = sim.get_translation(quadrotor_id)

    for frame in range(len(Coords)):
            
        dx = Coords[frame][0]
        dy = Coords[frame][1]
        dz = 0

        dispInFrame = np.array([dx, dy, dz])
        currentTranslation = np.array(sim.get_translation(quadrotor_id))

        sim.set_translation(dispInFrame + currentTranslation, quadrotor_id)

        if (sim.contact_test(quadrotor_id)):
            print("Collided at frame : ", frame)
            break

        if record:
        
            # get/save 1st person images
            # agent_observations = sim.get_sensor_observations()
            #save_observation(agent_observations, frame, "agent_")

            # move camera to 3rd person view
            sim.get_agent(0).scene_node.translation = np.array(
                static_3rd_person_camera_position
            )
            sim.get_agent(0).scene_node.rotation = ut.quat_to_magnum(
                ut.quat_look_at(
                    sim.get_translation(quadrotor_id), static_3rd_person_camera_position
                )
            )

            agent_observations = sim.get_sensor_observations()
            save_observation(agent_observations, frame, envNum, "3rdperson_")

            # reset the agent for the next step
            sim.get_agent(0).scene_node.translation = sim.get_translation(quadrotor_id)
            sim.get_agent(0).scene_node.rotation = sim.get_rotation(quadrotor_id)

    if record:
        MakeVideo(envNum)

    sim.close()
    del sim

def MakeVideo(envNum):
    # make a video from the frames
    fps = 60
    os.system(
        "ffmpeg -y -r "
        + str(fps)
        + " -f image2 -i CapturedFrames/Environment" + str(envNum) + "/3rdperson_rgba.%05d.png -f mp4 -q:v 0 -vcodec mpeg4 -r "
        + str(fps)
        + " CapturedFrames/Environment" + str(envNum) + "/3rdperson_rgba.mp4"
    )

    os.system(
        "ffmpeg -y -r "
        + str(fps)
        + " -f image2 -i CapturedFrames/Environment" + str(envNum) + "/3rdperson_depth.%05d.png -f mp4 -q:v 0 -vcodec mpeg4 -r "
        + str(fps)
        + " CapturedFrames/Environment" + str(envNum) + "/3rdperson_depth.mp4"
    )

if __name__ == "__main__":
    
    record = True
    seed = 1
    envNum = 1
    T_Horizon = 30
    path = "data/scene_datasets/allData/Ackermanville.glb"

    y = np.arange(0, T_Horizon, 1)
    Coords = [(0.05, 0.05) for j in y] 
    print(Coords)

    RunSimulation(envNum, record, path, Coords, seed)
