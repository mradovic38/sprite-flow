from abc import ABC, abstractmethod

import torch

from diff_eq.ode_sde import ODE, SDE


class Simulator(ABC):
    """
    Abstract base class for simulators.
    """
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Completes one step of the simulation.
        :param xt: state at time t, shape (bs, c, h, w)
        :param t: time, shape (bs, 1, 1, 1)
        :param dt: time change, shape(bs, 1, 1, 1)
        :return: nxt: state at time t + dt, shape (bs, c, h, w)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Simulate using discretization given by ts.
        :param x: initial state, shape(bs, c, h, w)
        :param ts: timesteps, shape (bs, nts, 1, 1, 1)
        :return: final state at time ts[-1], shape(bs, c, h, w)
        """
        nts = ts.shape[1]

        for t_idx in range(nts - 1):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)

        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Simulate with trajectory using discretization given by ts.
        :param x: initial state, shape(bs, c, h, w)
        :param ts: timesteps, shape (bs, nts, 1, 1, 1)
        :return: trajectory of xts over ts, shape(bs, c, h, w)
        """
        xs = [x.clone()]
        nts = ts.shape[1]

        for t_idx in range(nts - 1):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x)

        return torch.stack(xs, dim=1)


class EulerSimulator(Simulator):
    """
    Simulates an ODE using Euler method.
    """
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * dt


class EulerMaruyamaSimulator(Simulator):
    """
    Simulates an SDE using Euler-Maruyama method.
    """
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        return xt * self.sde.drift_coefficient(xt, t, **kwargs) * dt + \
            self.sde.diffusion_coefficient(xt, t, **kwargs) * torch.sqrt(dt) * torch.rand_like(xt)