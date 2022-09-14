#
import torch
import numpy as onp
from typing import List, Union, Tuple, cast
from models.gmm import ModelGMM
import math
import random

class ParameterSamplerHMC(object):
    R"""
    HMC parameter sampler.
    """
    #
    def __init__(self, model: ModelGMM, /) -> None:
        R"""
        Initialize the class.
        """
        # Pay attention optimizer is pseudo, we will not use it to update
        # parameters.
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)

        #
        self.shapes = [param.shape for param in self.model.parameters()]

    def velocity(
        self,
        stdv_or_stdvs: Union[float, List[float]], rng: torch.Generator,
        /,
    ) -> List[torch.Tensor]:
        R"""
        Sample random velocities from zero-mean Gaussian for all parameters.
        """
        # Formalize standard deviations.
        if isinstance(stdv_or_stdvs, list):
            #
            stdvs = stdv_or_stdvs
        else:
            #
            stdvs = [stdv_or_stdvs] * len(self.shapes)

        #
        return (
            [
                torch.randn(*shape, generator=rng).to(param.device) * stdv
                for (shape, param, stdv) in (
                    zip(self.shapes, self.model.parameters(), stdvs)
                )
            ]
        )

    def leapfrog(
        self,
        velos: List[torch.Tensor],
        /,
        *ARGS,
        deltas: List[float],
    ) -> None:
        R"""
        In-place leapfrog iteration.
        It should update `list(self.model.parameters())` as position $x$ in
        HMC.
        It should update `velos` as momentum $p$ in HMC.
        """
        self.optimizer.zero_grad()
        self.model.loss(*ARGS).backward()
        for pram, delta, vel in zip(self.model.parameters(), deltas, velos):
            vel -= torch.mul(delta/2, pram.grad.data)
        for pram, delta, vel in zip(self.model.parameters(), deltas, velos):
            pram.data += torch.mul(delta, vel)
        self.optimizer.zero_grad()
        self.model.loss(*ARGS).backward()
        for pram, delta, vel in zip(self.model.parameters(), deltas, velos):
            vel -= torch.mul(delta/2, pram.grad.data)


    def accept_or_reject(
        self,
        energy0: Tuple[float, float], energy1: Tuple[float, float],
        rng: torch.Generator,
    ) -> bool:
        R"""
        Given the energies `energy0` of the last sample and energies of new
        sample `energy1`, check if we should accept new sample.
        If True, we will accept new sample.
        If False, we will reject new sample, and repeat the last sample.
        """
        # YOU NEED TO FILL IN THIS PART.

        a = min(1, math.exp(energy0[0] + energy0[1] - energy1[0] - energy1[1]))
        if torch.rand(1, generator =rng) <= a:
            return True
        return False

    def sample(
        self,
        n: int,
        /,
        *ARGS,
        inits: List[torch.Tensor], stdv_or_stdvs: Union[float, List[float]],
        rng: torch.Generator, deltas: List[float], num_leapfrogs: int,
    ) -> List[List[torch.Tensor]]:
        R"""
        Sample from given parameters.
        """
        # Initialize buffer.
        samples = []
        potentials = []

        # Get initial sample.
        for (param, init) in zip(self.model.parameters(), inits):
            #
            param.data.copy_(init)
        with torch.no_grad():
            #
            nlf = self.model.energy(*ARGS).item()
        samples.append(
            [
                torch.clone(param.data.cpu())
                for param in self.model.parameters()
            ]
        )
        potentials.append(nlf)

        #
        print("{:s} {:s}".format("-" * 3, "-" * 6))
        num_accepts = 0
        for i in range(1, n + 1):
            # Sample a random velocity.
            # Get corresponding potential and kenetic energies.
            velos = self.velocity(stdv_or_stdvs, rng)
            nlf0 = potentials[-1]
            ke0 = sum(0.5 * torch.sum(velo ** 2).item() for velo in velos)

            # Update by multiple leapfrog steps to get a new sample.
            for _ in range(num_leapfrogs):
                #
                self.leapfrog(velos, *ARGS, deltas=deltas)
            with torch.no_grad():
                #
                nlf1 = self.model.energy(*ARGS).item()
            ke1 = sum(0.5 * torch.sum(velo ** 2).item() for velo in velos)

            # Metropolis-Hasting rejection sampling.
            accept_new = self.accept_or_reject((nlf0, ke0), (nlf1, ke1), rng)
            if accept_new:
                # Accept new samples.
                samples.append(
                    [
                        torch.clone(param.data.cpu())
                        for param in self.model.parameters()
                    ],
                )
                potentials.append(nlf1)
                print(
                    "{:>3d} {:>6s} {:>8s}"
                    .format(i, "Accept", "{:.6f}".format(nlf1)),
                )
            else:
                # Reject new samples.
                # Need to recover model parameters back to the last sample.
                samples.append(samples[-1])
                for (param, init) in zip(self.model.parameters(), samples[-1]):
                    #
                    param.data.copy_(init)
                potentials.append(nlf0)
                print(
                    "{:>3d} {:>6s} {:>8s}"
                    .format(i, "Reject", "{:.6f}".format(nlf0)),
                )
            num_accepts = num_accepts + int(accept_new)
        print("{:s} {:s}".format("-" * 3, "-" * 6))
        print("- Accept%: {:.1f}%".format(float(num_accepts) * 100 / float(n)))
        return samples