"""Clock
"""
from typing import Final
import glfw

# ---

# 1/60
DELTA: Final[float]= 0.01666666666666666667


# ---


class Clock:

    __slots__= ('ticks', 'delta', 'last_step', 'accumalate')

    def __init__(self) -> None:
        self.ticks: int= 0
        self.delta: float= DELTA
        self.last_step: float= 0.0
        self.accumalate: float= 0.0

    def update(self) -> None:
        """Update ticks and delta
        """
        current_step: float= glfw.get_time()
        elapsed: float= current_step - self.last_step

        self.last_step= current_step
        self.accumalate += elapsed

        while self.accumalate >= self.delta:
            self.accumalate -= self.delta
            self.ticks += 1.0

