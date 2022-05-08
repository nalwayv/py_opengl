"""Clock
"""
from dataclasses import dataclass
import glfw


# ---


@dataclass(eq= False, repr= False, slots= True)
class Clock:
    ticks: int= 0
    delta: float= 1.0 / 60.0
    _last_time_step: float= 0.0
    _accumalate: float= 0.0


    def update(self) -> None:
        """Update ticks and delta
        """
        current_time_step: float= glfw.get_time()
        elapsed: float= current_time_step - self._last_time_step

        self._last_time_step= current_time_step
        self._accumalate += elapsed

        while self._accumalate >= self.delta:
            self._accumalate -= self.delta
            self.ticks += 1.0