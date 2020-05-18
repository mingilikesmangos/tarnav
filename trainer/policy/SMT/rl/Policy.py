from rl.distributions import Categorical
import torch
import torch.nn as nn
from rl.utils import init


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

class PolicyWithBase(BasePolicy):
    def __init__(self, perception, action_space, num_stack=4):
        '''
            Args:
                base: A unit which of type ActorCriticModule
        '''
        super().__init__()
        self.perception_unit = perception
        # Make the critic
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self.perception_unit.output_size, 1))

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.perception_unit.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()


    def forward(self, inputs, states, masks): raise NotImplementedError

    def act(self, observations, masks, deterministic=False, mode='train'):

        x, pre_embedding = self.perception_unit.act(observations, masks, mode)
        value = self.critic_linear(x)
        dist = self.dist(x)

        if deterministic:
            # Select MAP/MLE estimate (depending on if we are feeling Bayesian)
            action = dist.mode()
        else:
            # Sample from trained posterior distribution
            action = dist.sample()


        self.probs = dist.probs
        action_log_probs = dist.log_probs(action)
        self.entropy = dist.entropy().mean()
        self.perplexity = torch.exp(self.entropy)

        return value, action, action_log_probs, pre_embedding

    def get_value(self, observations, masks, mode='train'):
        x = self.perception_unit(observations, masks, mode)
        value = self.critic_linear(x)
        return value

    def evaluate_actions(self, observations, masks, action, mode='train'):
        x = self.perception_unit(observations, masks, mode)
        value = self.critic_linear(x)
        actor_features = x
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

    def compute_intrinsic_losses(self, intrinsic_losses, inputs, states, masks, action, cache):
        losses = {}
        for intrinsic_loss in intrinsic_losses:
            if intrinsic_loss == 'activation_l2':
                assert 'residual' in cache, f'cache does not contain residual. it contains {cache.keys()}'  # (8*4) x 16 x 16
                diff = self.l2(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'activation_l1':
                assert 'residual' in cache, f'cache does not contain residual. it contains {cache.keys()}'
                diff = self.l1(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'perceptual_l1':  # only L1 since decoder
                assert 'residual' in cache, f'cache does not contain residual. it contains {cache.keys()}'
                act_teacher = self.decoder(inputs['taskonomy'])
                act_student = self.decoder(
                    cache['residual'])  # this uses a lot of memory... make sure that ppo_num_epoch=16
                diff = self.l1(act_teacher, act_student)
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'weight':
                pass
        return losses

class RecurrentPolicyWithBase(BasePolicy):
    def __init__(self, perception, action_space, internal_state_size=128):
        '''
            Args:
                base: A unit which of type ActorCriticModule
        '''
        super().__init__()
        self.perception_unit = perception
        self.gru = nn.GRUCell(input_size=internal_state_size, hidden_size=internal_state_size)

        # Make the critic
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self.perception_unit.output_size, 1))

        num_outputs = action_space.n
        self.dist = Categorical(internal_state_size, num_outputs)

        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()


    def forward(self, inputs, states, masks): raise NotImplementedError

    def act(self, observations, states, masks, deterministic=False, mode='train'):
        x, pre_embedding = self.perception_unit.act(observations, masks, mode)
        x = internal_states = self.gru(x, states * masks)
        value = self.critic_linear(x)
        dist = self.dist(x)

        if deterministic:
            # Select MAP/MLE estimate (depending on if we are feeling Bayesian)
            action = dist.mode()
        else:
            # Sample from trained posterior distribution
            action = dist.sample()


        self.probs = dist.probs
        action_log_probs = dist.log_probs(action)
        self.entropy = dist.entropy().mean()
        self.perplexity = torch.exp(self.entropy)

        return value, internal_states, action, action_log_probs, pre_embedding

    def get_value(self, observations, states, memory_masks=None, state_mask=None, mode='train'):
        x = self.perception_unit(observations, memory_masks, mode)
        if state_mask is None:
            x = self.gru(x, states)
        else:
            x = self.gru(x, states * state_mask)
        value = self.critic_linear(x)
        return value

    def evaluate_actions(self, observations, states, masks, action, state_masks, mode='train'):
        x = self.perception_unit(observations, masks, mode)
        x = internal_states = self.gru(x, states * state_masks)
        value = self.critic_linear(x)
        actor_features = x
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, internal_states

    def compute_intrinsic_losses(self, intrinsic_losses, inputs, states, masks, action, cache):
        losses = {}
        for intrinsic_loss in intrinsic_losses:
            if intrinsic_loss == 'activation_l2':
                assert 'residual' in cache, f'cache does not contain residual. it contains {cache.keys()}'  # (8*4) x 16 x 16
                diff = self.l2(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'activation_l1':
                assert 'residual' in cache, f'cache does not contain residual. it contains {cache.keys()}'
                diff = self.l1(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'perceptual_l1':  # only L1 since decoder
                assert 'residual' in cache, f'cache does not contain residual. it contains {cache.keys()}'
                act_teacher = self.decoder(inputs['taskonomy'])
                act_student = self.decoder(
                    cache['residual'])  # this uses a lot of memory... make sure that ppo_num_epoch=16
                diff = self.l1(act_teacher, act_student)
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'weight':
                pass
        return losses
