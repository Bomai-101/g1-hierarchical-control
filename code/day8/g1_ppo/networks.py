import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):

    def __init__(self, obs_dim=64, action_dim=29):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, action_dim)
        )

        self.log_std = nn.Parameter(
            torch.full(
                (action_dim,),
                -1.0
            )
        )
        
    def forward(self, obs):
        mu = self.net(obs)

        std = torch.exp(self.log_std)

        dist = Normal(mu, std)

        return dist


class Critic(nn.Module):

    def __init__(self, obs_dim=64):
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