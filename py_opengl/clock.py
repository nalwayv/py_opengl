"""
Clock
---
A Simple clock to get delta time and ticks based on glfw.get_time()
"""
import glfw
from dataclasses import dataclass

@dataclass(eq= False, repr= False, slots= True)
class Clock:
    ticks: int= 0
    delta: float= 1.0 / 60.0
    last_time_step: float= 0.0
    accumalate: float= 0.0


    def update(self) -> None:
        """Update clock
        """
        current_time_step: float= glfw.get_time()
        elapsed: float= current_time_step - self.last_time_step

        self.last_time_step= current_time_step
        self.accumalate += elapsed

        while self.accumalate >= self.delta:
            self.accumalate -= self.delta
            self.ticks += 1.0