import torch


class RolloutBuffer:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(
        self,
        observation,
        action,
        reward,
        value,
        log_prob,
        done
    ):

        self.observations.append(
            observation.detach().cpu()
        )

        self.actions.append(
            action.detach().cpu()
        )

        self.rewards.append(
            float(reward)
        )

        self.values.append(
            value.detach().cpu()
        )

        self.log_probs.append(
            log_prob.detach().cpu()
        )

        self.dones.append(
            float(done)
        )

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)

if __name__ == "__main__":

    buffer = RolloutBuffer()

    obs = torch.randn(64)
    action = torch.randn(29)
    reward = 1.5
    value = torch.tensor([0.3], requires_grad=True)
    log_prob = torch.tensor(-2.0, requires_grad=True)
    done = False

    buffer.add(
        obs,
        action,
        reward,
        value,
        log_prob,
        done
    )

    print("Buffer length:", len(buffer))
    print("Observation shape:", buffer.observations[0].shape)
    print("Action shape:", buffer.actions[0].shape)
    print("Reward:", buffer.rewards[0])
    print("Value requires_grad:", buffer.values[0].requires_grad)
    print("Log prob requires_grad:", buffer.log_probs[0].requires_grad)

    buffer.clear()

    print("Buffer length after clear:", len(buffer))