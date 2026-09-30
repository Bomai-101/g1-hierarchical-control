import torch
import torch.nn as nn
from torch.distributions import Normal
from g1_control.config.balance import ACTION_DIM, ACTOR_LOG_STD, OBS_DIM



class Actor(nn.Module):

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        log_std=ACTOR_LOG_STD,
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(
                256,
                action_dim
            )
        )

        # Start from exact zero residual policy.
        nn.init.zeros_(
            self.net[-1].weight
        )

        nn.init.zeros_(
            self.net[-1].bias
        )

        self.log_std = nn.Parameter(
            torch.full(
                (action_dim,),
                float(log_std)
            ),
            requires_grad=False
        )

    def forward(
        self,
        obs
    ):

        mu = self.net(obs)

        std = torch.exp(
            self.log_std
        )

        return Normal(
            mu,
            std
        )


class Critic(nn.Module):

    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, 1)
        )

    def forward(self, obs):
        value = self.net(obs)

        return value


if __name__ == "__main__":

    actor = Actor()
    critic = Critic()

    obs = torch.randn(64)

    dist = actor(obs)

    action = dist.sample()

    log_prob = dist.log_prob(action).sum()

    value = critic(obs)

    print("Observation shape:", obs.shape)
    print("Action shape:", action.shape)
    print("Action:", action)
    print("Log probability:", log_prob)
    print("Value:", value)
