#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo import Net, Policy

import torchvision.models as models

class PointNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid="pointgoal_with_gps_compass",
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        resnet_baseplanes=32,
        backbone="resnet50",
        normalize_visual_inputs=False,
    ):
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                goal_sensor_uuid=goal_sensor_uuid,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
            ),
            action_space.n,
        )
        
class PointNavResNetPolicy_ns(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid="pointgoal_with_gps_compass",
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        resnet_baseplanes=32,
        backbone="resnet50",
        normalize_visual_inputs=False,
    ):
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                goal_sensor_uuid=goal_sensor_uuid,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
            ),
            action_space.n - 1,
        )

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        make_backbone=None,
        normalize_visual_inputs=False,
    ):
        super().__init__()

        if "panoramic_rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["panoramic_rgb"].shape[2]
            spatial_size = observation_space.spaces["panoramic_rgb"].shape[0] // 2
        else:
            self._n_input_rgb = 0

        if "panoramic_depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["panoramic_depth"].shape[2]
            spatial_size = observation_space.spaces["panoramic_depth"].shape[0] // 2
        else:
            self._n_input_depth = 0

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 1024
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial*2,
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, input):
        if self.is_blind:
            return None

        #x = torch.cat(input_list, dim=1)
        x = F.avg_pool2d(input, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x

import time
TIME_DEBUG = True
def log_time(prev_time, log):
    print("[TIME] ", log, time.time() - prev_time)
    return time.time()

class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32

        #self._n_input_goal =
        self.num_category = 50
        #self.tgt_embeding = nn.Linear(self.num_category, 32)
        self._n_input_goal = 0

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape)*2, hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_tgt_encoding(self, goal_observations):
        goal_onehot = torch.eye(self.num_category)[goal_observations[:,0,0].long()].to(goal_observations.device)
        return self.tgt_embeding(goal_onehot)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        B = observations['panoramic_rgb'].shape[0]
        input_list = [observations['panoramic_rgb'].permute(0,3,1,2)/255.0,
                      observations['panoramic_depth'].permute(0,3,1,2)]
        curr_obs = torch.cat(input_list,1)

        goal_obs = observations['objectgoal'].permute(0,3,1,2)
        #goal_obs = observations['target_goal'].permute(0,3,1,2)
        batched_obs = torch.cat([curr_obs, goal_obs[:,:4]],0)

        feats = self.visual_encoder(batched_obs)
        curr_feats, target_feats = feats.split(B)

        #tgt_encoding = self.get_tgt_encoding(goal_obs[:,-1])
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )
        feats = self.visual_fc(torch.cat((curr_feats.view(B,-1),target_feats.view(B,-1)),1))
        x = [feats, prev_actions]

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states

class ResnetTargetDrivenPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
    ):
        super().__init__(
            ResnetTargetDrivenNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                action_space = action_space,
            ),
            action_space.n,
        )

class ResnetTargetDrivenNet(Net):

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, action_space):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32

        self.num_category = 50
        self._n_input_goal = 0

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action

        self.visual_encoder = ResnetSiameseNetwork()

        self.state_encoder = RNNStateEncoder(
            self._hidden_size + rnn_input_size,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        B = observations['panoramic_rgb'].shape[0]
        #input_list = [observations['panoramic_rgb'].permute(0,3,1,2)/255.0,
        #              observations['panoramic_depth'].permute(0,3,1,2)]
        input_list = [observations['panoramic_rgb'].permute(0,3,1,2)/255.0]
        curr_obs = torch.cat(input_list,1)

        goal_obs = observations['target_goal'].permute(0, 3, 1, 2)/255.0
        batched_obs = torch.cat([curr_obs, goal_obs[:,:3]],0)

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        perception_embed = self.visual_encoder(batched_obs, B)

        x = [perception_embed]

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states

class ResnetSiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Resnet Layer
        self.resnet_ = models.resnet50(pretrained=True)
        weight = self.resnet_.conv1.weight.clone()
        self.resnet_.conv1 = nn.Conv2d(4,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.resnet_.conv1.weight[:,:3]= weight
        nn.init.kaiming_normal_(self.resnet_.conv1.weight[:,3], nn.init.calculate_gain("relu"))
        self.resnet_ = torch.nn.Sequential(*(list(self.resnet_.children())[:-1]))

        # Siamese & Merge layer
        self.visual_fc = nn.Sequential(
                nn.Linear(
                    2048*2, 512
                ),
                nn.ReLU(True),
            )


    def forward(self, batched_obs, B):

        feats = self.resnet_(batched_obs)
        curr_feats, goal_feats = feats.split(B)

        feats = self.visual_fc(
            torch.cat((curr_feats.view(B, -1), goal_feats.view(B, -1)), 1)
            )

        return feats
