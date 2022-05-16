"""Window
"""
from typing import Any, Callable, Optional

import glfw


# ---


class GLWindowError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class GlWindow:

    __slots__= ('width', 'height', 'title', 'window')

    def __init__(self, width: int, height: int, title: str):
        """
        Raises
        ---
        GLWindowError
            failed to init glfw window
        """
        self.width: int= width
        self.height: int= height
        self.title: str= title
        self.window: Optional[Any]= glfw.create_window(
            self.width,
            self.height,
            self.title,
            None,
            None
        )

        if not self.window:
            raise GLWindowError('failed to init glfw window')

    def set_window_resize_callback(self, cbfun: Callable[[Any, float, float], None]) -> None:
        """Set glfw window callback for window resize
        """
        glfw.set_window_size_callback(self.window, cbfun)

    def should_close(self) -> bool:
        """Close window
        """
        return True if glfw.window_should_close(self.window) else False

    def center_screen_position(self) -> None:
        """Center glfw window
        """
        video= glfw.get_video_mode(glfw.get_primary_monitor())

        x: float= (video.size.width // 2) - (self.width // 2)
        y: float= (video.size.height // 2) - (self.height // 2)

        glfw.set_window_pos(self.window, x, y)


    def get_mouse_pos(self) -> tuple[float, float]:
        """Return current mouse screen position
        """
        return glfw.get_cursor_pos(self.window)


    def get_mouse_state(self, button: int) -> tuple[int, int]:
        """Get glfw mouse button state plus return button number passed in

        GLFW_RELEASE = 0
        
        GLFW_PRESS = 1
        """
        return (button, glfw.get_mouse_button(self.window, button))


    def get_key_state(self, key: int) -> tuple[int, int]:
        """Return glfw key state plus key number passed in

        GLFW_RELEASE = 0

        GLFW_PRESS = 1
        """
        return (key, glfw.get_key(self.window, key))