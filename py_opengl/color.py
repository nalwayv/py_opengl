"""Color RGB
"""

from dataclasses import dataclass
from py_opengl import glm

@dataclass(eq=False, repr=False, slots=True)
class Color:
    _data: int = 0xFF

    @staticmethod
    def from_rgba(red, green, blue, alpha) -> 'Color':
        c = Color()
        c.set_red(red)
        c.set_green(green)
        c.set_blue(blue)
        c.set_alpha(alpha)
        return c

    def get_red(self) -> int:
        return 0xFF & (self._data)

    def get_green(self) -> int:
        return 0xFF & (self._data >> 8)

    def get_blue(self) -> int:
        return 0xFF & (self._data >> 16)

    def get_alpha(self) -> int:
        return 0xFF & (self._data >> 32)

    def set_red(self, value: int) -> None: 
        self._data = (self._data & 0xFFFFFF00) | (glm.clamp(value, 0, 255))

    def set_green(self, value: int) -> None:
        self._data = (self._data & 0xFFFF00FF) | (glm.clamp(value, 0, 255) << 8)

    def set_blue(self, value: int) -> None:
        self._data = (self._data & 0xFF00FFFF) | (glm.clamp(value, 0, 255) << 16)

    def set_alpha(self, value: int) -> None:
        self._data = (self._data & 0x00FFFFFF) | (glm.clamp(value, 0, 255) << 32)

    def get_data(self) -> tuple[int, int, int, int]:
        return (
            self.get_red(),
            self.get_green(),
            self.get_blue(),
            self.get_alpha()
        )

    def get_data_norm(self) -> tuple[float, float, float, float]:
        return (
            float(self.get_red() / 255),
            float(self.get_green() / 255),
            float(self.get_blue() / 255),
            float(self.get_alpha() / 255)
        )