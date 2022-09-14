#
import torch
from typing import Iterable, Optional, Callable


class Optimizer(torch.optim.Optimizer):
    r"""
    Optimizer.
    """
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.
        """
        # Save necessary attributes.
        self.lr = lr
        self.weight_decay = weight_decay

        # Super call.
        torch.optim.Optimizer.__init__(self, parameters, dict())

    @torch.no_grad()
    def prev(self, /) -> None:
        r"""
        Operations before compute the gradient.
        PyTorch has design problem of compute Nesterov SGD gradient.
        PyTorch team avoid this problem by using an approximation of Nesterov
        SGD gradient.
        Also, using closure can also solve the problem, but it maybe a bit
        complicated for this homework.
        In our case, function is provided as auxiliary function for simplicity.
        It is called before `.backward()`.
        This function is only used for Nesterov SGD gradient.
        """
        # Do nothing.
        pass

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.
        """
        #
        ...


class SGD(Optimizer):
    r"""
    SGD.
    """
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.
        """
        #
        Optimizer.__init__(self, parameters, lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.
        """
        # Traverse parameters of each groups.
        for group in self.param_groups:
            #
            for parameter in group['params']:
                # Get gradient without weight decaying.
                if parameter.grad is None:
                    #
                    continue
                else:
                    #
                    gradient = parameter.grad
                    gradient.add_(parameter.data, alpha = 2 * self.weight_decay)
                # Apply weight decay.
                # YOU SHOULD FILL IN THIS FUNCTION
                ...

                # Gradient Decay.
                parameter.data.add_(gradient, alpha=-self.lr)
        return None


class Momentum(SGD):
    R"""
    Momentum.
    """
    #
    def __init__(
            self,
            /,
            parameters: Iterable[torch.nn.parameter.Parameter],
            *,
            lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.
        """
        #
        SGD.__init__(self, parameters, lr=lr, weight_decay=weight_decay)
        self.v  = [None] * len(self.param_groups[0]['params'])

    @torch.no_grad()
    def step(
            self,
            /,
            closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.
        """
        # Traverse parameters of each groups.
        RHO = 0.9
        for group in self.param_groups:
            #
            for i, parameter in enumerate(group['params']):
                # Get gradient without weight decaying.
                if parameter.grad is None:
                    #
                    continue
                else:
                    #
                    gradient = parameter.grad
                    gradient.add_(parameter.data, alpha = 2 * self.weight_decay)
                    if self.v[i] is None:
                        self.v[i] = torch.zeros(gradient.size(), device = gradient.device, dtype = gradient.dtype)
                    else:
                        self.v[i].mul_(RHO)
                        self.v[i].add_(gradient, alpha = self.lr)


                # Apply weight decay.
                # YOU SHOULD FILL IN THIS FUNCTION
                ...

                # Gradient Decay.
                parameter.data.add_(self.v[i], alpha=-1)
        return None


class Nesterov(SGD):
    R"""
    Nesterov.
    """
    #
    def __init__(
            self,
            /,
            parameters: Iterable[torch.nn.parameter.Parameter],
            *,
            lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.
        """
        #
        SGD.__init__(self, parameters, lr=lr, weight_decay=weight_decay)
        self.v = [None] * len(self.param_groups[0]['params'])

    @torch.no_grad()
    def prev(self, /) -> None:
        RHO = 0.9
        for group in self.param_groups:
            for i, parameter in enumerate(group['params']):
                if self.v[i] is not None:
                    self.v[i].mul_(RHO)
                    parameter.data.add_(self.v[i], alpha=-1)

    def step(
            self,
            /,
            closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.
        """
        # Traverse parameters of each groups.
        RHO = 0.9
        for group in self.param_groups:
            #
            for i, parameter in enumerate(group['params']):
                # Get gradient without weight decaying.
                if parameter.grad is None:
                    #
                    continue
                else:
                    #
                    gradient = parameter.grad
                    gradient.add_(parameter.data, alpha = 2 * self.weight_decay)
                    if self.v[i] is None:
                        self.v[i] = torch.zeros(gradient.size(), device=gradient.device, dtype=gradient.dtype)
                    else:
                        parameter.data.add_(self.v[i], alpha=1)
                        self.v[i].add_(gradient, alpha=self.lr)

                # Apply weight decay.
                # YOU SHOULD FILL IN THIS FUNCTION
                ...

                # Gradient Decay.
                parameter.data.add_(self.v[i], alpha=-1)


class Adam(SGD):
    R"""
    Adam.
    """
    #
    def __init__(
            self,
            /,
            parameters: Iterable[torch.nn.parameter.Parameter],
            *,
            lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.
        """
        #
        SGD.__init__(self, parameters, lr=lr, weight_decay=weight_decay)
        self.m1 = [None] * len(self.param_groups[0]['params'])
        self.m2 = [None] * len(self.param_groups[0]['params'])

    @torch.no_grad()
    def step(
            self,
            /,
            closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.
        """
        # Traverse parameters of each groups.
        BETA1 = 0.9
        BETA2 = 0.999
        EPSILON = 1e-8
        for group in self.param_groups:
            for i, parameter in enumerate(group['params']):
                # Get gradient without weight decaying.
                if parameter.grad is None:
                    #
                    continue
                else:
                    #
                    gradient = parameter.grad
                    gradient.add_(parameter.data, alpha = 2 * self.weight_decay)
                    if self.m1[i] is None:
                        self.m1[i] = torch.zeros(gradient.size(), device=gradient.device, dtype=gradient.dtype)
                        self.m2[i] = torch.zeros(gradient.size(), device=gradient.device, dtype=gradient.dtype)
                    else:
                        self.m1[i].mul_(BETA1)
                        self.m1[i].add_(gradient, alpha= 1 - BETA1)
                        self.m2[i].mul_(BETA2)
                        r = torch.square(gradient)
                        self.m2[i].add_(r, alpha=1 - BETA2)

                # Apply weight decay.
                # YOU SHOULD FILL IN THIS FUNCTION
                ...

                # Gradient Decay.
                u1 = torch.div(self.m1[i], 1 - BETA1)
                u2 = torch.div(self.m2[i], 1 - BETA2)
                su2 = torch.sqrt(u2)
                su2.add_(EPSILON)

                parameter.data.add_(torch.div(u1, su2), alpha=-self.lr)
        return None