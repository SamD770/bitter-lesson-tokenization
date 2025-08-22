import random
from typing import Tuple

class DownsampleRateScheduler:
    """
    Base class for using a variable downsample rate throughout the training loop.
    DownsampleRateScheduler.step() is called at the start of every effective batch.
    DownsampleRateScheduler.val_step() is called at the start of every validation step.
    """
    def step(self) -> Tuple[float, float]:
        """
        Returns a tuple containing: the downsample rate and target_downsample_rate for the current step.
        """
        raise NotImplementedError()

    def val_step(self) -> float:
        """
        Returns a tuple containing: the downsample rate and target_downsample_rate for the current validationstep.
        """
        raise NotImplementedError()



class DefaultDownsampleRateScheduler(DownsampleRateScheduler):
    """
    Default downsample rate scheduler that uses the default downsample rate from passing None into model.forward().
    """
    def __init__(self, downsample_rate_target):
        self.downsample_rate_target = downsample_rate_target

    def step(self):
        return None, self.downsample_rate_target

    def val_step(self):
        return None, self.downsample_rate_target


class RandomDownsampleRateScheduler(DownsampleRateScheduler):
    def __init__(self, min_downsample_rate, max_downsample_rate, val_downsample_rate):
        self.min_downsample_rate = min_downsample_rate
        self.max_downsample_rate = max_downsample_rate
        self.val_downsample_rate = val_downsample_rate

    def step(self):
        downsample_rate = random.uniform(self.min_downsample_rate, self.max_downsample_rate)
        return downsample_rate

    def val_step(self):
        return self.val_downsample_rate, self.val_downsample_rate


class RandomDownsampleRateSchedulerWithInit(DownsampleRateScheduler):
    """
    Allows for the configuring of a deterministic downsample rate for the first step, for debugging CUDA OOMing from a high downsample rate.
    """
    def __init__(self, min_downsample_rate, max_downsample_rate, init_downsample_rate=1):
        self.min_downsample_rate = min_downsample_rate
        self.max_downsample_rate = max_downsample_rate
        self.first_step = True
        self.init_downsample_rate = init_downsample_rate

    def step(self, step_count):

        if self.first_step:
            self.first_step = False
            downsample_rate = self.init_downsample_rate
        else:
            downsample_rate = random.uniform(self.min_downsample_rate, self.max_downsample_rate)
        
        return downsample_rate, downsample_rate

    def val_step(self):
        return self.init_downsample_rate, self.init_downsample_rate