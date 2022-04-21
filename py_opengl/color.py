"""Color RGBA
"""

from dataclasses import dataclass
from py_opengl import maths


# ---


@dataclass(eq= False, repr= False, slots= True)
class Color:
    rgba: int= 0xFF

    @staticmethod
    def from_rgba(red: int, green: int, blue: int, alpha: int) -> 'Color':
        c= Color()
        c.set_red(red)
        c.set_green(green)
        c.set_blue(blue)
        c.set_alpha(alpha)
        return c

    def get_red(self) -> int:
        return 0xFF & (self.rgba)

    def get_green(self) -> int:
        return 0xFF & (self.rgba >> 8)

    def get_blue(self) -> int:
        return 0xFF & (self.rgba >> 16)

    def get_alpha(self) -> int:
        return 0xFF & (self.rgba >> 24)

    def set_red(self, value: int) -> None: 
        self.rgba= (self.rgba & 0xFFFFFF00) | (maths.clampi(value, 0, 255))

    def set_green(self, value: int) -> None:
        self.rgba= (self.rgba & 0xFFFF00FF) | (maths.clampi(value, 0, 255) << 8)

    def set_blue(self, value: int) -> None:
        self.rgba= (self.rgba & 0xFF00FFFF) | (maths.clampi(value, 0, 255) << 16)

    def set_alpha(self, value: int) -> None:
        self.rgba= (self.rgba & 0x00FFFFFF) | (maths.clampi(value, 0, 255) << 24)

    def get_data(self) -> tuple[int, int, int, int]:
        return (
            self.get_red(),
            self.get_green(),
            self.get_blue(),
            self.get_alpha()
        )

    def get_data_norm(self) -> tuple[float, float, float, float]:
        inv= 1.0 / 255.0
        return (
            float(self.get_red()) * inv,
            float(self.get_green()) * inv,
            float(self.get_blue()) * inv,
            float(self.get_alpha()) * inv
        )