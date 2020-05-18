from trainer.policy.SMT.rl.distributions import Categorical
import torch
import torch.nn as nn
from trainer.policy.SMT.rl.utils import init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.utils import CategoricalNet, Flatten
class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError

    def act(self, observations, masks, deterministic=False):
        ''' TODO: find out what these do '''
        raise NotImplementedError
        return value, action, action_log_probs, states

    def get_value(self, observations, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError
        return value

    def evaluate_actions(self, observations, masks, action):
        ''' TODO: find out what these do '''
        raise NotImplementedError
        return value, action_log_probs, dist_entropy, states

from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo import Net, Policy
from trainer.policy.resnet.resnet_policy import ResNetEncoder
from trainer.policy.SMT.model.perception import Perception


class SMTPolicy(Policy):
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
        cfg=None
    ):
        super().__init__(
            SMTNet(
                observation_space=observation_space,
                action_space=action_space,
                goal_sensor_uuid=goal_sensor_uuid,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                cfg=cfg
            ),
            action_space.n,
        )


    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states



class SMTNet(Net):
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
            cfg
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self.action_dim = action_space.n
        self.prev_action_embedding = nn.Embedding(self.action_dim + 1, 32)
        self._n_prev_action = 32

        # self._n_input_goal =
        self.num_category = 50
        self.tgt_embeding = nn.Linear(self.num_category, 32)
        self._n_input_goal = 0

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action + 1
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        self.visual_fc = nn.Sequential(
            nn.Linear(
                np.prod(self.visual_encoder.output_shape) * 2, hidden_size
            ),
            nn.ReLU(True),
        )
        self.embedding_shape = (cfg.NUM_PROCESSES, self._hidden_size + rnn_input_size)
        self.perception_unit = Perception(cfg, (self.visual_encoder, self.prev_action_embedding, self.visual_fc))
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
        goal_onehot = torch.eye(self.num_category)[goal_observations[:, 0, 0].long()].to(goal_observations.device)
        return self.tgt_embeding(goal_onehot)

    def eval_embed_network(self):
        self.visual_encoder.eval()
        self.visual_fc.eval()
        self.prev_action_embedding.eval()

    def embedd(self, observations, prev_actions=None, masks=None):
        B = observations['panoramic_rgb'].shape[0]
        input_list = [observations['panoramic_rgb'].permute(0, 3, 1, 2) / 255.0,
                      observations['panoramic_depth'].permute(0, 3, 1, 2)]
        curr_obs = torch.cat(input_list, 1)

        goal_obs = observations['target_goal'].permute(0, 3, 1, 2)
        batched_obs = torch.cat([curr_obs, goal_obs[:, :4]], 0)

        feats = self.visual_encoder(batched_obs)
        curr_feats, target_feats = feats.split(B)
        if prev_actions is not None and masks is not None:
            prev_actions_embedd = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().squeeze(-1)
            )
        else:
            prev_actions_embedd = self.prev_action_embedding(torch.zeros((B),dtype=torch.long).to(feats.device))
        feats = self.visual_fc(torch.cat((curr_feats.view(B, -1), target_feats.view(B, -1)), 1))
        x = [feats, prev_actions_embedd, torch.exp(-observations['step']/10).to(feats.device).view(B,-1)]

        x = torch.cat(x, dim=1)
        return x

    def act(self, observations, rnn_hidden_states, prev_actions, masks):
        x, pre_embedding = self.perception_unit.act(observations['embeddings'], observations['memory_masks'])
        #x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states, pre_embedding

    def forward(self, observation_memory, rnn_hidden_states, masks):
        # need perception here
        x = self.perception_unit(observation_memory, masks)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states

