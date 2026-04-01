from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


class Simulator(ABC):
    """With batchsize."""

    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """
        Description:

            Takes one simulation step
        inputs:
            xt: state at time t, shape (batch_size, dim)
            t: time, shape ()
            dt: time, shape ()
        outputs:
            nxt: state at time t + dt
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Description:

            Simulates using the discretization gives by ts
        inputs:
            x_init: initial state at time ts[0], shape (batch_size, dim)
            ts: timesteps, shape (nts,)
        outputs:
            x_fina: final state at time ts[-1], shape (batch_size, dim)
        """
        for t_idx in range(len(ts) - 1):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            # pyrefly: ignore [bad-assignment]
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Description:

            Simulates using the discretization gives by ts
        inputs:
            x_init: initial state at time ts[0], shape (bs, dim)
            ts: timesteps, shape (num_timesteps,)
        outputs:
            xs: trajectory of xts over ts, shape (batch_size, num_timesteps, dim)
        """
        xs = [x.clone()]
        for t_idx in tqdm(range(len(ts) - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            # pyrefly: ignore [bad-assignment]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)


# without batchsize
# class Simulator(ABC):
#    """without batchsize"""
#
#    @abstractmethod
#    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
#        pass
#
#    @torch.no_grad()
#    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
#        for t_idx in range(len(ts) - 1):
#            t = ts[t_idx]
#            h = ts[t_idx + 1] - ts[t_idx]
#            x = self.step(x, t, h)
#        return x
#
#    @torch.no_grad()
#    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
#        xs = [x.clone()]
#        for t_idx in tqdm(range(len(ts) - 1)):
#            t = ts[t_idx]
#            h = ts[t_idx + 1] - ts[t_idx]
#            x = self.step(x, t, h)
#            xs.append(x.clone())
#        return torch.stack(xs, dim=1)
