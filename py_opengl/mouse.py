"""Mouse
"""
from enum import Enum, auto
from py_opengl import window

# ---


class MouseState(Enum):
    PRESSED= auto()
    RELEASED= auto()
    HELD= auto()
    DEFAULT= auto()


# ---


class Mouse:

    __slots__= ('states', 'win')

    def __init__(self, win: window.GlWindow) -> None:
        self.states: list[int]= [0xFF] * 3
        self.win: window.GlWindow|None= win

    def _set_current_state_at(self, key: int, value: int) -> None:
        """Set glfw mouse button number stored current state
        """
        self.states[key]= (self.states[key] & 0xFFFFFF00) | value


    def _set_previous_state_at(self, key: int, value: int) -> None:
        """Set glfw mouse button number stored previous state
        """
        self.states[key]= (self.states[key] & 0xFFFF00FF) | (value << 8)

    def _get_current_state_at(self, key: int) -> int:
        """Get glfw mouse button number stored current state
        """
        return 0xFF & self.states[key]

    def _get_previous_state_at(self, key: int) -> int:
        """Get glfw mouse button number stored previous state
        """
        return 0xFF & (self.states[key] >> 8)

    def _get_state(self, glfw_mouse_state: int) -> MouseState:
        """Mouse button pressed

        use GlWindow function 'get_mouse_state' for glfw_mouse_state
        """
        current= self.win.get_mouse_state(glfw_mouse_state)
        key= glfw_mouse_state
        result= MouseState.DEFAULT

        if key < 3:
            prev = self._get_current_state_at(key)
            self._set_previous_state_at(key, prev)
            self._set_current_state_at(key, current)

            if self._get_previous_state_at(key) == 0:
                if self._get_current_state_at(key) == 0:
                    result= MouseState.DEFAULT
                else:
                    # pressed
                    result= MouseState.PRESSED
            else:
                if self._get_current_state_at(key) == 0:
                    # released
                    result= MouseState.RELEASED
                else:
                    # held
                    result= MouseState.HELD

        return result

    def is_button_held(self, mouse: int) -> bool:
        """Return true if mouse button is held
        """
        return self._get_state(mouse) is MouseState.HELD

    def is_button_pressed(self, mouse: int) -> bool:
        """Return true if mouse button is just being pressed
        """
        return self._get_state(mouse) is MouseState.PRESSED

    def is_button_released(self, mouse: int) -> bool:
        """Return true if mouse button is released
        """
        return self._get_state(mouse) is MouseState.RELEASED

