from typing import Optional, Type

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from utils.vis_utils import observations_to_image, append_text_to_image
import cv2
from gym.spaces.dict_space import Dict as SpaceDict

def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="CustomObjectNavRLEnv")
class CustomObjectNavEnv(habitat.RLEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self.stuck = 0
        self.progress = 0
        self.closest = 50
        self._previous_measure = None
        self._previous_action = None
        self.time_t = 0
        super().__init__(self._core_env_config, dataset)
        self.observation_space = SpaceDict(
            {
                'panoramic_rgb': self.habitat_env._task.sensor_suite.observation_spaces.spaces['panoramic_rgb'],
                'panoramic_depth': self.habitat_env._task.sensor_suite.observation_spaces.spaces['panoramic_depth'],
                'objectgoal': self.habitat_env._task.sensor_suite.observation_spaces.spaces['objectgoal']
            }
        )

    def reset(self):
        self._previous_action = None
        self.time_t = 0
        self.stuck = 0
        self.progress = 0
        self.closest = 50
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        self.obs = observations
        self.info = None
        self.total_reward = 0
        return self.process_obs(observations)

    def process_obs(self, obs):
        return {'panoramic_rgb': obs['panoramic_rgb'],
                'panoramic_depth': obs['panoramic_depth'],
                'objectgoal': obs['objectgoal']}
    def step(self, action):
        #action['action'] += 1
        self._previous_action = action
        
        obs, reward, done, self.info = super().step(action)
        
        self.time_t += 1
        self.info['length'] = self.time_t * done
        #print(self.info)
        #print(self.habitat_env.sim.previous_step_collided)
        #print("agent state : {0}, {1} #####################".format(
        #    self.habitat_env.sim.get_agent_state().position,
        #    self.habitat_env.sim.get_agent_state().rotation)
        #)
        self.obs = obs
        self.total_reward += reward
        return self.process_obs(obs), reward, done, self.info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward_type = self._rl_config.REWARD_SHAPE
        
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        self.progress =  self._previous_measure - current_measure
        
        if reward_type == "dense":
            reward += self.progress*(self._rl_config.REWARD_RATIO)
           
        if reward_type == "dist":
            dist = self.habitat_env.get_metrics()['distance_to_goal']
            if dist < self.closest:
                covered = self.closest - dist
                self.closest = dist
                if covered <10:
                    reward += covered*(self._rl_config.REWARD_RATIO)
        

        if abs(self.progress) < 0.01:
            self.stuck += 1
        else:
            self.stuck = 0

        self._previous_measure = current_measure

        if self._episode_success():
            reward = self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self.stuck > 40:
            done = True

        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def render(self, mode='rgb'):
        info = self.get_info(None) if self.info is None else self.info
        img = observations_to_image(self.obs, info, mode='panoramic')
        str_action = ''
        if self._previous_action is not None:
            action_list = ["STOP", "MOVE_FORWARD", 'TURN_LEFT', 'TURN_RIGHT']
            str_action = action_list[self._previous_action['action']]
        dist = self.habitat_env.get_metrics()['distance_to_goal']
        category = self.current_episode.object_category
        img = append_text_to_image(img, 't: %03d, r: %f a: %s, dist: %.2f cat: %s'%(self.time_t,self.total_reward, str_action, dist, category))
        if mode == 'rgb' or mode == 'rgb_array':
            return img
        elif mode == 'human':
            cv2.imshow('render', img)
            cv2.waitKey(1)
            return img
        return super().render(mode)
    
@baseline_registry.register_env(name="CustomObjectNavRLEnv-val")
class CustomObjectNavEnvVal(habitat.RLEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self.stuck = 0
        self.progress = 0
        self.closest = 50
        self._previous_measure = None
        self._previous_action = None
        self.time_t = 0
        super().__init__(self._core_env_config, dataset)
        self.observation_space = SpaceDict(
            {
                'panoramic_rgb': self.habitat_env._task.sensor_suite.observation_spaces.spaces['panoramic_rgb'],
                'panoramic_depth': self.habitat_env._task.sensor_suite.observation_spaces.spaces['panoramic_depth'],
                'objectgoal': self.habitat_env._task.sensor_suite.observation_spaces.spaces['objectgoal']
            }
        )

    def reset(self):
        self._previous_action = None
        self.time_t = 0
        self.stuck = 0
        self.progress = 0
        self.closest = 50
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        self.obs = observations
        self.info = None
        self.total_reward = 0
        return self.process_obs(observations)

    def process_obs(self, obs):
        return {'panoramic_rgb': obs['panoramic_rgb'],
                'panoramic_depth': obs['panoramic_depth'],
                'objectgoal': obs['objectgoal']}
    def step(self, action):
        #action['action'] += 1
        self._previous_action = action
        
        obs, reward, done, self.info = super().step(action)
        
        self.time_t += 1
        self.info['length'] = self.time_t * done
        #print(self.info)
        #print(self.habitat_env.sim.previous_step_collided)
        #print("agent state : {0}, {1} #####################".format(
        #    self.habitat_env.sim.get_agent_state().position,
        #    self.habitat_env.sim.get_agent_state().rotation)
        #)
        self.obs = obs
        self.total_reward += reward
        return self.process_obs(obs), reward, done, self.info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward_type = self._rl_config.REWARD_SHAPE
        
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        self.progress =  self._previous_measure - current_measure
        
        if reward_type == "dense":
            reward += self.progress*(self._rl_config.REWARD_RATIO)
           
        if reward_type == "dist":
            dist = self.habitat_env.get_metrics()['distance_to_goal']
            if dist < self.closest:
                covered = self.closest - dist
                self.closest = dist
                if covered <10:
                    reward += covered*(self._rl_config.REWARD_RATIO)
        

        if abs(self.progress) < 0.01:
            self.stuck += 1
        else:
            self.stuck = 0

        self._previous_measure = current_measure

        if self._episode_success():
            reward = self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        #if self.stuck > 40:
        #    done = True

        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def render(self, mode='rgb'):
        info = self.get_info(None) if self.info is None else self.info
        img = observations_to_image(self.obs, info, mode='panoramic')
        str_action = ''
        if self._previous_action is not None:
            action_list = ["STOP", "MOVE_FORWARD", 'TURN_LEFT', 'TURN_RIGHT']
            str_action = action_list[self._previous_action['action']]
        dist = self.habitat_env.get_metrics()['distance_to_goal']
        category = self.current_episode.object_category
        img = append_text_to_image(img, 't: %03d, r: %f a: %s, dist: %.2f cat: %s'%(self.time_t,self.total_reward, str_action, dist, category))
        if mode == 'rgb' or mode == 'rgb_array':
            return img
        elif mode == 'human':
            cv2.imshow('render', img)
            cv2.waitKey(1)
            return img
        return super().render(mode)

if __name__ == '__main__':
    def filter_fn(episode):
        if episode.info['geodesic_distance'] < 5.0:
            return True
        else:
            return False

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
        task_config.TASK.CUSTOM_OBJECT_GOAL_SENSOR.GOAL_SPEC = "OBJECT_IMG"
        task_config.TASK.PANORAMIC_SENSOR = habitat.Config()
        task_config.TASK.PANORAMIC_SENSOR.TYPE = 'PanoramicRGBSensor'
        task_config.TASK.PANORAMIC_SENSOR.WIDTH = task_config.SIMULATOR.RGB_SENSOR.WIDTH
        task_config.TASK.PANORAMIC_SENSOR.HEIGHT = task_config.SIMULATOR.RGB_SENSOR.HEIGHT
        task_config.TASK.PANORAMIC_DEPTH_SENSOR = task_config.SIMULATOR.DEPTH_SENSOR.clone()
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.TYPE = 'PanoramicDepthSensor'
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.WIDTH = task_config.SIMULATOR.DEPTH_SENSOR.WIDTH
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.HEIGHT = task_config.SIMULATOR.DEPTH_SENSOR.HEIGHT
        return task_config


    from habitat_baselines.common.baseline_registry import baseline_registry
    from env_utils.object_nav_dataset import CustomObjectNavDatasetV1
    from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
    from habitat_baselines.config.default import get_config
    import numpy as np
    config = get_config('configs/ddppo_objectnav.yaml')
    #config.defrost()
    #config.TASK_CONFIG.DATASET.CONTENT_SCENES = ['1LXtFkjw3qL']
    #config.freeze()
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET, **{'filter_fn': filter_fn})
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    config.defrost()
    config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG)
    config.freeze()

    env  = CustomObjectNavEnv(config, dataset)
    obs = env.reset()
    img = env.render('rgb')
    while True:
        for i in env.current_episode.shortest_paths[0]:
            img = env.render('rgb')
            cv2.imshow('render', img[:, :, [2, 1, 0]])
            key = cv2.waitKey(0)
            if key == ord('s'): action = 0
            elif key == ord('w'): action = 1
            elif key == ord('a'): action = 2
            elif key == ord('d'): action = 3
            elif key == ord('q'): break
            else:
                action = i
                if i is None :
                    action = 0
            obs, reward, done, info = env.step({'action':action})
            if done:
                obs = env.reset()
                break

