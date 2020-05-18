#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, RGBSensor, DepthSensor
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)
import quaternion as q
import time

def make_panoramic(left, front, right, back):
    return np.concatenate([left, front, right, back],1)

@registry.register_sensor(name="CustomVisTargetSensor")
class CustomVisTargetSensor(Sensor):


    def __init__(
        self, sim, config: Config, dataset: Dataset, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dataset = dataset
        use_depth = 'DEPTH_SENSOR' in self._sim.config.AGENT_0.SENSORS
        use_rgb = 'RGB_SENSOR' in self._sim.config.AGENT_0.SENSORS
        self.channel = use_depth + 3 * use_rgb
        self.height = self._sim.config.RGB_SENSOR.HEIGHT if use_rgb else self._sim.config.DEPTH_SENSOR.HEIGHT
        self.width = self._sim.config.RGB_SENSOR.WIDTH if use_rgb else self._sim.config.DEPTH_SENSOR.WIDTH
        self.curr_episode_id = -1
        self.curr_scene_id = ''
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "target_goal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=1.0, shape=(self.height, self.width*4, self.channel), dtype=np.float32)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ) -> Optional[int]:

        episode_id = episode.episode_id
        scene_id = episode.scene_id
        if (self.curr_episode_id != episode_id) or (self.curr_scene_id != scene_id):
            position = episode.goals[0].position
            euler = [0, 2 * np.pi * np.random.rand(), 0]
            rotation = q.from_rotation_vector(euler)
            obs = self._sim.get_observations_at(position,rotation)
            rgb_array = make_panoramic(obs['rgb_left'], obs['rgb'], obs['rgb_right'], obs['rgb_back'])/255.
            depth_array = make_panoramic(obs['depth_left'], obs['depth'], obs['depth_right'], obs['depth_back'])

            self.goal_obs = np.concatenate([rgb_array, depth_array],2)
            self.curr_episode_id = episode_id
            self.curr_scene_id = scene_id
        return self.goal_obs

