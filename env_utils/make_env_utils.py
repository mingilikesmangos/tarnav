#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Type, Union

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from utils.visdommonitor import VisdomMonitor
import os

def filter_fn(episode):
    #if 'difficulty' in episode.info.keys():
    #    if episode.info['difficulty'] == 'easy' :
    #        return True
    #    else : return False
    #if episode.info['geodesic_distance'] < 4.0 and episode.info['geodesic_distance'] > 1.5:
    if episode.info['geodesic_distance'] < 7 and episode.info['geodesic_distance'] > 4.0:
    #if episode.info['geodesic_distance'] < 10.0 and episode.info['geodesic_distance'] > 7:
    #if episode.info['geodesic_distance'] < 10.0 and episode.info['geodesic_distance'] > 1.5:
        #if episode.object_category in ['bed', 'chair', 'stool', 'table', 'cabinet', 'tv_monitor']:
        #if episode.object_category in ['chair']:
        if episode.object_category in ['table']:
        #if episode.object_category in ['picture', 'window']:
            return True

    else : return False

import numpy as np
def add_panoramic_camera(task_config):
    task_config.SIMULATOR.RGB_SENSOR_LEFT = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_LEFT.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_LEFT.ORIENTATION = [0, 0.5 * np.pi, 0]
    task_config.SIMULATOR.RGB_SENSOR_LEFT.ANGLE = "left"
    task_config.SIMULATOR.RGB_SENSOR_RIGHT = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_RIGHT.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_RIGHT.ORIENTATION = [0, -0.5 * np.pi, 0]
    task_config.SIMULATOR.RGB_SENSOR_RIGHT.ANGLE = "right"
    task_config.SIMULATOR.RGB_SENSOR_BACK = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_BACK.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_BACK.ORIENTATION = [0, np.pi, 0]
    task_config.SIMULATOR.RGB_SENSOR_BACK.ANGLE = "back"
    task_config.SIMULATOR.AGENT_0.SENSORS += ['RGB_SENSOR_LEFT', 'RGB_SENSOR_RIGHT', 'RGB_SENSOR_BACK']

    task_config.SIMULATOR.DEPTH_SENSOR_LEFT = task_config.SIMULATOR.DEPTH_SENSOR.clone()
    task_config.SIMULATOR.DEPTH_SENSOR_LEFT.TYPE = "PanoramicPartDepthSensor"
    task_config.SIMULATOR.DEPTH_SENSOR_LEFT.ORIENTATION = [0, 0.5 * np.pi, 0]
    task_config.SIMULATOR.DEPTH_SENSOR_LEFT.ANGLE = "left"
    task_config.SIMULATOR.DEPTH_SENSOR_RIGHT = task_config.SIMULATOR.DEPTH_SENSOR.clone()
    task_config.SIMULATOR.DEPTH_SENSOR_RIGHT.TYPE = "PanoramicPartDepthSensor"
    task_config.SIMULATOR.DEPTH_SENSOR_RIGHT.ORIENTATION = [0, -0.5 * np.pi, 0]
    task_config.SIMULATOR.DEPTH_SENSOR_RIGHT.ANGLE = "right"
    task_config.SIMULATOR.DEPTH_SENSOR_BACK = task_config.SIMULATOR.DEPTH_SENSOR.clone()
    task_config.SIMULATOR.DEPTH_SENSOR_BACK.TYPE = "PanoramicPartDepthSensor"
    task_config.SIMULATOR.DEPTH_SENSOR_BACK.ORIENTATION = [0, np.pi, 0]
    task_config.SIMULATOR.DEPTH_SENSOR_BACK.ANGLE = "back"
    task_config.SIMULATOR.AGENT_0.SENSORS += ['DEPTH_SENSOR_LEFT', 'DEPTH_SENSOR_RIGHT', 'DEPTH_SENSOR_BACK']

    task_config.TASK.CUSTOM_OBJECT_GOAL_SENSOR = habitat.Config()
    task_config.TASK.CUSTOM_OBJECT_GOAL_SENSOR.TYPE = 'CustomObjectSensor'
    task_config.TASK.CUSTOM_VISTARGET_SENSOR = habitat.Config()
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.TYPE = 'CustomVisTargetSensor'
    task_config.TASK.CUSTOM_OBJECT_GOAL_SENSOR.GOAL_SPEC = "OBJECT_IMG"
    task_config.TASK.PANORAMIC_SENSOR = habitat.Config()
    task_config.TASK.PANORAMIC_SENSOR.TYPE = 'PanoramicRGBSensor'
    task_config.TASK.PANORAMIC_SENSOR.WIDTH = task_config.SIMULATOR.RGB_SENSOR.WIDTH
    task_config.TASK.PANORAMIC_SENSOR.HEIGHT = task_config.SIMULATOR.RGB_SENSOR.HEIGHT
    task_config.TASK.PANORAMIC_DEPTH_SENSOR = task_config.SIMULATOR.DEPTH_SENSOR.clone()
    task_config.TASK.PANORAMIC_DEPTH_SENSOR.TYPE = 'PanoramicDepthSensor'
    task_config.TASK.PANORAMIC_DEPTH_SENSOR.WIDTH = task_config.SIMULATOR.DEPTH_SENSOR.WIDTH
    task_config.TASK.PANORAMIC_DEPTH_SENSOR.HEIGHT = task_config.SIMULATOR.DEPTH_SENSOR.HEIGHT

    if "STOP" not in task_config.TASK.POSSIBLE_ACTIONS:
        task_config.TASK.SUCCESS.TYPE = "Success_woSTOP"
    task_config.TASK.SUCCESS.SUCCESS_DISTANCE = task_config.TASK.SUCCESS_DISTANCE
    return task_config

def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]], rank: int, kwargs
) -> Union[Env, RLEnv]:
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET, **kwargs
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(rank)
    #env = VisdomMonitor(env,
    #                    directory = config.VIDEO_DIR,
    #                    video_callable = lambda x : x % config.VIS_INTERVAL == 0,
    #                    uid = str(rank)
    #                    )
    return env


def construct_envs(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created.

    Returns:
        VectorEnv object created according to specification.
    """

    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [env_class for _ in range(num_processes)]

    # for debug!
    #config.defrost()
    #print('***!!!!!!!!!!!!!!!!**************debug code not deleted')
    #config.TASK_CONFIG.DATASET.CONTENT_SCENES = ['ZMojNkEp431']
    #config.freeze()
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, **{'filter_fn': filter_fn})
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
        scenes = scenes[:32]
    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_processes:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

        random.shuffle(scenes)


    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)
    print('Total Process : %d '%num_processes)
    for i, s in enumerate(scene_splits):
        print('proc %d :'%i, s)
    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config = add_panoramic_camera(task_config)

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )

        #task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(zip(configs, env_classes, range(num_processes), [{'filter_fn': filter_fn}]*num_processes))
        )
    )
    return envs
