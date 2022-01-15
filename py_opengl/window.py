"""GLFW Window
"""
from dataclasses import dataclass
from typing import (Any, Optional, Callable)

import glfw

# notes
# https://www.glfw.org/docs/latest/input_guide.html#input_keyboard

class GlWindowError(Exception):
    """Gl Window Error

    Parameters
    ---
    Exception
    """
    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class GlWindow:
    window: Optional[Any] = None
    width: int = 0
    height: int = 0
    title: str = "glfw_window"

    def __post_init__(self):
        self.window = glfw.create_window(
            self.width,
            self.height,
            self.title,
            None,
            None
        )

        if not self.window:
            raise GlWindowError('failed to init glfw window')

    def set_window_resize_callback(self, cb: Callable[[Any, float, float], None]) -> None:
        """Set glfw window callback for window resize

        Parameters
        ---
        cb : Callable[[Any, float, float], None]
            callback function that takes current glfw window and its current width and height
        """
        glfw.set_window_size_callback(self.window, cb)

    def should_close(self) -> bool:
        """Close window

        Returns
        ---
        bool
        """
        return True if glfw.window_should_close(self.window) else False


    def center_screen_position(self) -> None:
        """Center glfw window
        """
        video = glfw.get_video_mode(glfw.get_primary_monitor())

        x: float = (video.size.width // 2) - (self.width // 2)
        y: float = (video.size.height // 2) - (self.height // 2)

        glfw.set_window_pos(self.window, x, y)


    def get_mouse_pos(self) -> tuple[float, float]:
        """Return current mouse screen position

        Returns
        ---
        tuple[float, float]
        """
        return glfw.get_cursor_pos(self.window)


    def get_mouse_state(self, button: int) -> tuple[int, int]:
        """Get glfw mouse button state

        Parameters
        ---
        button : int
            glfw mouse button

        Returns
        ---
        tuple[int, int]
            mouse button passed and its current glfw state

            GLFW_RELEASE = 0

            GLFW_PRESS = 1
        """
        return (button, glfw.get_mouse_button(self.window, button))


    def get_key_state(self, key: int) -> tuple[int, int]:
        """Return glfw key state

        Parameters
        ---
        key : int
            glfw key

        Returns
        ---
        tuple[int, int]
            key passed and its current glfw state

            GLFW_RELEASE = 0

            GLFW_PRESS = 1
        """
        return (key, glfw.get_key(self.window, key))