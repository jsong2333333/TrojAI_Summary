import math

import gymnasium as gym
import torch
import torch.nn as nn

from trojai_rlgen.modelgen.architectures.architecture_utils import linear_w_relu, ModdedResnet18

"""
Models are primarily adaptations of https://github.com/lcswillems/rl-starter-files/blob/master/model.py
"""


class TrojaiDRLBackbone(nn.Module):
    """Base class for TrojAI DRL models"""

    def __init__(self, embedding, actor, critic):
        nn.Module.__init__(self)
        self.state_emb = embedding  # Define state embedding
        self.actor = actor  # Define actor's model
        self.critic = critic  # Define critic's model
        self.value = 0

    def forward(self, obs):
        agent_dir = obs['direction'].long()
        obs = obs['image']
        x = self.state_emb(obs.float())
        x = x.reshape(x.shape[0], -1)
        x = torch.concat([x, agent_dir], dim=1)
        x_act = self.actor(x)
        x_crit = self.critic(x)
        self.value = x_crit.squeeze(1)
        return x_act

    def value_function(self):
        return self.value

    def args_dict(self):
        raise NotImplementedError("Should be implemented in subclass")


class FCModel(TrojaiDRLBackbone):
    """Fully-connected actor-critic model with shared embedding"""

    def __init__(self, obs_space, action_space, linear_embedding_dims=(512, 256), actor_linear_mid_dims=(64, 32),
                 critic_linear_mid_dims=(64, 32)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param linear_embedding_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the linear embedding
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """
        self.obs_space = obs_space
        if isinstance(self.obs_space, gym.spaces.Dict):
            flattened_dims = int(math.prod(self.obs_space['image'].shape))
        else:
            flattened_dims = int(math.prod(self.obs_space.shape))
        self.action_space = action_space
        self.linear_embedding_dims = linear_embedding_dims
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims

        # +1 because we concat direction information to embedding
        self.state_embedding_size = linear_embedding_dims[-1] + 1
        embedding = linear_w_relu([flattened_dims] + [d for d in linear_embedding_dims])
        embedding.insert(0, nn.Flatten())  # put a flattening layer in front
        actor_dims = [self.state_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [self.state_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            embedding,
            linear_w_relu(actor_dims, end_relu=False),
            linear_w_relu(critic_dims, end_relu=False)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'linear_embedding_dims': self.linear_embedding_dims,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }


class CNNModel(TrojaiDRLBackbone):
    """Simple actor-critic model with CNN embedding"""

    def __init__(self, obs_space, action_space, channels=(16, 32, 64), actor_linear_mid_dims=(144,),
                 critic_linear_mid_dims=(144,)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param channels: (iterable) Sequence of 3 integers representing the number of numbers of channels to use for the
            CNN embedding
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """
        if len(channels) != 3:
            raise ValueError("'channels' must be a tuple or list of length 3")

        self.obs_space = obs_space
        self.action_space = action_space
        self.channels = channels
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims

        c1, c2, c3 = channels
        image_embedding_size = 4 * 4 * c3 + 1  # +1 because we concat direction information to embedding
        image_conv = nn.Sequential(
            nn.Conv2d(3, c1, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(c1, c2, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(c2, c3, (2, 2)),
            nn.ReLU()
        )
        actor_dims = [image_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [image_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            image_conv,
            linear_w_relu(actor_dims, end_relu=False),
            linear_w_relu(critic_dims, end_relu=False)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'channels': self.channels,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }


class ImageACModel(TrojaiDRLBackbone):
    """Simple CNN Actor-Critic model designed for MiniGrid. Assumes 48x48 grayscale or RGB images."""

    def __init__(self, obs_space, action_space, channels=(8, 16, 32), actor_linear_mid_dims=(144,),
                 critic_linear_mid_dims=(144,)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Technically unused
            for this model, but stored both for consistency between models and to be used for later reference if needed.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param channels: (iterable) Sequence of 3 integers representing the number of numbers of channels to use for the
            CNN embedding
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """

        self.obs_space = obs_space
        self.action_space = action_space
        self.channels = channels
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims
        self.image_size = 48  # this is the size of image this CNN was designed for

        num_channels = obs_space['image'].shape[0]
        c1, c2, c3 = channels
        image_embedding_size = 3 * 3 * c3 + 1  # +1 because we concat direction information to embedding
        image_conv = nn.Sequential(
            nn.Conv2d(num_channels, c1, (3, 3), stride=3),
            nn.ReLU(),
            nn.Conv2d(c1, c2, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, (3, 3), stride=2),
            nn.ReLU()
        )
        actor_dims = [image_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [image_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            image_conv,
            linear_w_relu(actor_dims, end_relu=False),
            linear_w_relu(critic_dims, end_relu=False)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'channels': self.channels,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }


class ResNetACModel(TrojaiDRLBackbone):
    """Actor-Critic model with ResNet18 embedding designed for MiniGrid. Assumes 112x112 RGB images."""

    def __init__(self, obs_space, action_space, actor_linear_mid_dims=(512,), critic_linear_mid_dims=(512,)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Technically unused
            for this model, but stored both for consistency between models and to be used for later reference if needed.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims

        image_embedding_size = 512 + 1  # +1 because we concat direction information to embedding
        embedding = ModdedResnet18()
        actor_dims = [image_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [image_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            embedding,
            linear_w_relu(actor_dims, end_relu=False),
            linear_w_relu(critic_dims, end_relu=False)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }
