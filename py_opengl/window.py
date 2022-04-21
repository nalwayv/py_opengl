"""Window
"""
from dataclasses import dataclass
from typing import Any, Callable

import glfw


# ---


class GlWindowError(Exception):
    """Gl Window Error

    Parameters
    ---
    Exception
    """
    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq= False, repr= False, slots= True)
class GlWindow:
    window: Any|None= None
    width: int= 0
    height: int= 0
    title: str= "glfw_window"

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

    def set_window_resize_callback(self, cbfun: Callable[[Any, float, float], None]) -> None:
        """Set glfw window callback for window resize

        Parameters
        ---
        cbfun : Callable[[Any, float, float], None]

        """
        glfw.set_window_size_callback(self.window, cbfun)

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

        x: float= (video.size.width // 2) - (self.width // 2)
        y: float= (video.size.height // 2) - (self.height // 2)

        glfw.set_window_pos(self.window, x, y)


    def get_mouse_pos(self) -> tuple[float, float]:
        """Return current mouse screen position

        Returns
        ---
        tuple[float, float]
        """
        return glfw.get_cursor_pos(self.window)


    def get_mouse_state(self, button: int) -> tuple[int, int]:
        """Get glfw mouse button state plus return button number passed in

        GLFW_RELEASE = 0
        
        GLFW_PRESS = 1

        Parameters
        ---
        button : int

        Returns
        ---
        tuple[int, int]

        """
        return (button, glfw.get_mouse_button(self.window, button))


    def get_key_state(self, key: int) -> tuple[int, int]:
        """Return glfw key state plus key number passed in

        GLFW_RELEASE = 0

        GLFW_PRESS = 1

        Parameters
        ---
        key : int

        Returns
        ---
        tuple[int, int]

        """
        return (key, glfw.get_key(self.window, key))